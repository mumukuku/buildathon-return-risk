"""
RiskGuard synthetic data generator
===================================

Generates two datasets for the AI Risk Manager buildathon track:

1. orders.csv        -> order/return-level data for the Return-Risk Scorer
2. customers.csv      -> customer-level data, including shared-identifier
                          fields (device / address / payment fingerprint)
                          for the Abuse-Ring Sentinel

IMPORTANT (read this before trusting any metric downstream):
This is SYNTHETIC data. No real merchant, customer, or transaction data
is used anywhere in this project. Ground-truth labels come from a known
generative rule (defined below), which lets us honestly report precision/
recall against something -- but it also means the numbers describe how
well the model recovers OUR rule, not real-world fraud. We are explicit
about this in the README and metrics report. The rule is designed to be
realistic (grounded in publicly documented return-fraud patterns such as
wardrobing and bracketing) but imperfectly separable -- i.e. we inject
noise so that a perfect classifier is impossible, which is what real
fraud data looks like.

Customer behavior types (hidden from the model, used only to drive
generation and for our own auditing):
    - normal              (70%): occasional genuine returns
    - occasional_returner  (18%): returns more often, still genuine
    - serial_abuser         (7%): wardrobing / bracketing patterns
    - ring_member           (5%): shares device/address/payment fingerprint
                                   with other "distinct" accounts

Two important design choices to avoid common ML mistakes:
    1. NO LEAKAGE: rolling features (customer's historical return rate,
       order count, chargeback count) are computed using only orders that
       happened strictly BEFORE the current one, per customer, in time
       order.
    2. Hidden 'behavior_type' is NEVER written into orders.csv as a
       feature -- only into customers.csv for our own auditing /
       ring-detector evaluation. Using it as a model feature would be
       cheating (it's essentially the label generator).
"""

import numpy as np
import pandas as pd
import hashlib
import uuid
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_CUSTOMERS = 6000
SIM_DAYS = 365
START_DATE = datetime(2025, 1, 1)

CATEGORIES = ["apparel", "footwear", "electronics", "mobile", "beauty", "home", "accessories"]
CATEGORY_BASE_RETURN_RATE = {
    "apparel": 0.30, "footwear": 0.26, "accessories": 0.18,
    "beauty": 0.10, "home": 0.14, "electronics": 0.12, "mobile": 0.09,
}
CATEGORY_AVG_VALUE = {
    "apparel": 1400, "footwear": 2200, "accessories": 900,
    "beauty": 700, "home": 1800, "electronics": 6500, "mobile": 14000,
}
PAYMENT_METHODS = ["UPI", "card", "COD", "wallet"]
RETURN_REASONS = ["size_issue", "changed_mind", "damaged", "not_as_described", "wrong_item", "no_longer_needed"]

BEHAVIOR_TYPES = ["normal", "occasional_returner", "serial_abuser", "ring_member"]
BEHAVIOR_WEIGHTS = [0.70, 0.18, 0.07, 0.05]


def make_fingerprint(seed_str: str) -> str:
    """Stable pseudo-hash standing in for a device/address/payment fingerprint."""
    return hashlib.sha1(seed_str.encode()).hexdigest()[:12]


def build_customers(n: int) -> pd.DataFrame:
    behavior_types = rng.choice(BEHAVIOR_TYPES, size=n, p=BEHAVIOR_WEIGHTS)
    account_age_days = rng.integers(5, 1500, size=n)

    customer_ids = [f"CUST{100000+i}" for i in range(n)]

    # Most customers get their own unique device/address/payment fingerprint.
    device_fp = [make_fingerprint(f"dev-{cid}") for cid in customer_ids]
    address_fp = [make_fingerprint(f"addr-{cid}") for cid in customer_ids]
    payment_fp = [make_fingerprint(f"pay-{cid}") for cid in customer_ids]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "behavior_type": behavior_types,           # HIDDEN from model features
        "account_age_days": account_age_days,
        "device_fingerprint": device_fp,
        "address_fingerprint": address_fp,
        "payment_fingerprint": payment_fp,
        "ring_id": [None] * n,                      # filled in below
    })

    # --- Build shared-identifier structures ---
    # 1) Genuine benign sharing (confounder): e.g. families / hostel mates
    #    sharing an address. This is NOT abuse -- the ring detector must
    #    learn to tell this apart from real abuse rings.
    #    IMPORTANT: some benign groups ALSO share a payment fingerprint
    #    (e.g. a family using one shared credit card) -- this is deliberate
    #    overlap with the ring signal below, so "shares_payment" alone
    #    cannot trivially separate the two classes. A real detector has to
    #    combine identifier signals with behavior, not just identifier
    #    sharing.
    benign_pool = df.sample(frac=0.06, random_state=1).index.tolist()
    rng.shuffle(benign_pool)
    bi = 0
    benign_group_num = 0
    while bi < len(benign_pool) - 1:
        group_size = rng.integers(2, 4)  # 2-3 people sharing an address
        group = benign_pool[bi: bi + group_size]
        if len(group) < 2:
            break
        shared_addr = make_fingerprint(f"benign-addr-{benign_group_num}")
        df.loc[group, "address_fingerprint"] = shared_addr
        if rng.random() < 0.35:  # ~35% of benign groups also share a payment method
            shared_family_payment = make_fingerprint(f"benign-pay-{benign_group_num}")
            df.loc[group, "payment_fingerprint"] = shared_family_payment
        bi += group_size
        benign_group_num += 1

    # 2) Abuse rings: ring_member customers are grouped into rings of 3-9.
    #    To avoid making this trivially separable (a naive rule would just
    #    check "shares device AND payment"), rings vary in how much they
    #    share, simulating operators with different levels of caution:
    #      - ~55% share BOTH device and payment (obvious, careless rings)
    #      - ~25% share ONLY device (payment method varies -- e.g. different
    #        stolen/synthetic cards on the same phone)
    #      - ~20% share ONLY payment (device varies -- e.g. same stolen card
    #        used across different devices)
    ring_members = df.index[df["behavior_type"] == "ring_member"].tolist()
    rng.shuffle(ring_members)
    ri = 0
    ring_num = 0
    while ri < len(ring_members) - 1:
        group_size = min(rng.integers(3, 10), len(ring_members) - ri)
        if group_size < 3:
            break
        group = ring_members[ri: ri + group_size]
        sharing_mode = rng.choice(["both", "device_only", "payment_only"], p=[0.55, 0.25, 0.20])
        if sharing_mode in ("both", "device_only"):
            shared_dev = make_fingerprint(f"ring-dev-{ring_num}")
            df.loc[group, "device_fingerprint"] = shared_dev
        if sharing_mode in ("both", "payment_only"):
            shared_pay = make_fingerprint(f"ring-pay-{ring_num}")
            df.loc[group, "payment_fingerprint"] = shared_pay
        df.loc[group, "ring_id"] = f"RING{ring_num}"
        ri += group_size
        ring_num += 1

    return df


def simulate_orders(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cust in customers.iterrows():
        btype = cust["behavior_type"]

        # Order volume depends loosely on account age.
        n_orders = max(1, int(rng.poisson(lam=max(1, cust["account_age_days"] / 90))))
        n_orders = min(n_orders, 40)

        order_days = np.sort(rng.integers(0, SIM_DAYS, size=n_orders))

        # Rolling state used to compute pre-order historical features (no leakage).
        past_orders = 0
        past_returns = 0
        past_abusive_returns = 0
        past_chargebacks = 0

        for od in order_days:
            order_date = START_DATE + timedelta(days=int(od))
            category = rng.choice(CATEGORIES)
            base_val = CATEGORY_AVG_VALUE[category]
            order_value = float(np.round(rng.lognormal(mean=np.log(base_val), sigma=0.5), 2))
            payment_method = rng.choice(PAYMENT_METHODS, p=[0.42, 0.28, 0.20, 0.10])
            delivery_days = int(rng.integers(1, 8))
            order_hour = int(rng.integers(0, 24))
            is_weekend = int(order_date.weekday() >= 5)

            # --- historical features computed BEFORE this order is resolved ---
            hist_return_rate = past_returns / past_orders if past_orders > 0 else 0.0
            hist_abusive_rate = past_abusive_returns / past_orders if past_orders > 0 else 0.0
            hist_orders = past_orders
            hist_chargebacks = past_chargebacks
            price_vs_cat_avg = order_value / base_val

            # --- return probability depends on category + behavior type ---
            base_rate = CATEGORY_BASE_RETURN_RATE[category]
            type_multiplier = {
                "normal": 1.0, "occasional_returner": 1.6,
                "serial_abuser": 2.4, "ring_member": 1.4,
            }[btype]
            return_prob = min(0.95, base_rate * type_multiplier)
            is_returned = rng.random() < return_prob

            return_reason = None
            days_to_return = None
            is_abusive = 0

            if is_returned:
                days_to_return = int(rng.integers(1, 31))

                # reason distribution shifts by behavior type
                if btype == "serial_abuser":
                    reason_probs = [0.30, 0.35, 0.10, 0.15, 0.05, 0.05]
                elif btype == "ring_member":
                    reason_probs = [0.15, 0.20, 0.20, 0.25, 0.10, 0.10]
                else:
                    reason_probs = [0.22, 0.28, 0.20, 0.10, 0.10, 0.10]
                return_reason = rng.choice(RETURN_REASONS, p=reason_probs)

                # --- ground-truth abuse rule (documented, imperfectly separable) ---
                abuse_score = 0.0
                if btype == "serial_abuser":
                    abuse_score += 0.55
                if btype == "ring_member":
                    abuse_score += 0.30
                if return_reason in ("changed_mind", "size_issue") and price_vs_cat_avg > 1.3:
                    abuse_score += 0.20          # wardrobing signature: pricier item, vague reason
                if days_to_return >= 25:
                    abuse_score += 0.15          # returned right at the deadline
                if hist_abusive_rate > 0.3:
                    abuse_score += 0.20          # repeat offender
                if payment_method == "COD" and return_reason == "not_as_described":
                    abuse_score += 0.10          # correlates with claim-without-return fraud patterns
                abuse_score += rng.normal(0, 0.18)  # noise -> imperfect separability, by design

                is_abusive = int(abuse_score > 0.55)
                if is_abusive:
                    past_abusive_returns += 1
                    if rng.random() < 0.25:
                        past_chargebacks += 1  # some abusive returns escalate to a chargeback later

                past_returns += 1

            past_orders += 1

            rows.append({
                "order_id": f"ORD{uuid.uuid4().hex[:10]}",
                "customer_id": cust["customer_id"],
                "order_date": order_date.strftime("%Y-%m-%d"),
                "category": category,
                "order_value": order_value,
                "payment_method": payment_method,
                "delivery_days": delivery_days,
                "order_hour": order_hour,
                "is_weekend": is_weekend,
                "account_age_days_at_order": max(0, cust["account_age_days"] - (SIM_DAYS - od)),
                "hist_orders_before": hist_orders,
                "hist_return_rate_before": round(hist_return_rate, 4),
                "hist_abusive_return_rate_before": round(hist_abusive_rate, 4),
                "hist_chargebacks_before": hist_chargebacks,
                "price_vs_category_avg": round(price_vs_cat_avg, 3),
                "is_returned": int(is_returned),
                "return_reason": return_reason,
                "days_to_return": days_to_return,
                "device_fingerprint": cust["device_fingerprint"],
                "address_fingerprint": cust["address_fingerprint"],
                "payment_fingerprint": cust["payment_fingerprint"],
                # label -- only meaningful / used for training on rows where is_returned == 1
                "is_abusive_return": is_abusive,
                # audit-only column, NEVER a model feature:
                "_behavior_type": btype,
            })

    return pd.DataFrame(rows)


def main():
    customers = build_customers(N_CUSTOMERS)
    orders = simulate_orders(customers)

    customers.to_csv("data/customers.csv", index=False)
    orders.to_csv("data/orders.csv", index=False)

    returned = orders[orders["is_returned"] == 1]
    print(f"customers: {len(customers)}")
    print(f"orders:    {len(orders)}")
    print(f"returns:   {len(returned)}  ({len(returned)/len(orders):.1%} of orders)")
    print(f"abusive returns: {returned['is_abusive_return'].sum()}  "
          f"({returned['is_abusive_return'].mean():.1%} of returns)")
    print(f"ring members: {(customers['behavior_type']=='ring_member').sum()}  "
          f"across {customers['ring_id'].nunique()} rings")
    print("\nBehavior type breakdown:")
    print(customers["behavior_type"].value_counts())


if __name__ == "__main__":
    main()
