"""
Synthetic data generator for a subscription (SaaS / Telecom) business.

Why synthetic data instead of a public churn file?
------------------------------------------------
Public churn datasets are cross-sectional: one row per customer, no history, no way to
build a time-aware model. This generator simulates the *process* that produces churn:

    latent customer traits -> monthly behaviour -> monthly churn hazard -> cancellation

Churn is never sampled directly from a label column. Each month, every active customer
draws a Bernoulli event whose probability is a logistic function of things a real analyst
would actually see: contract type, support-ticket pressure, usage trend, late payments,
satisfaction, tenure, price pressure and an unobserved per-customer frailty term. That
frailty is deliberately hidden from the modelling table, which caps achievable AUC at a
realistic level instead of the suspicious 0.99 that leaky synthetic data usually gives.

Output (deliberately messy - cleaning is part of the project):
    data/raw/customers.csv   one row per customer
    data/raw/plans.csv       plan catalogue (tier x contract)
    data/raw/usage.csv       customer-month engagement
    data/raw/payments.csv    customer-month billing
    data/raw/support.csv     ticket-level support events
    data/raw/status.csv      churn outcome per customer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from config import (
    EARLIEST_SIGNUP,
    N_CUSTOMERS,
    N_MONTHS,
    RAW_DIR,
    SEED,
    WINDOW_START,
)

# --------------------------------------------------------------------------------------
# Static reference data
# --------------------------------------------------------------------------------------
PLAN_TIERS = [
    # name,       price,  data limit,   features
    ("Starter", 19.99, "50 GB", "Core app;Email support"),
    ("Basic", 34.99, "150 GB", "Core app;Email support;Mobile app"),
    ("Standard", 59.99, "500 GB", "Core app;Mobile app;Analytics;Priority email"),
    ("Plus", 89.99, "1 TB", "Core app;Mobile app;Analytics;Integrations;API access"),
    ("Pro", 129.99, "2.5 TB", "Core app;Mobile app;Advanced analytics;Integrations;API access;SSO;Phone support"),
    ("Enterprise", 249.99, "Unlimited", "Core app;Mobile app;Advanced analytics;Integrations;API access;SSO;Dedicated CSM;Custom SLA"),
]
DATA_LIMIT_GB = [50, 150, 500, 1000, 2500, 10_000]
CONTRACTS = [("Month-to-month", 1.00), ("One year", 0.92), ("Two year", 0.85)]

SEGMENTS = ["Consumer", "SMB", "Enterprise"]
SEGMENT_P = [0.58, 0.32, 0.10]

# Tier preference by segment (rows: segment, cols: tier)
TIER_BY_SEGMENT = np.array(
    [
        [0.30, 0.28, 0.22, 0.12, 0.06, 0.02],  # Consumer
        [0.05, 0.15, 0.28, 0.28, 0.18, 0.06],  # SMB
        [0.00, 0.03, 0.10, 0.22, 0.35, 0.30],  # Enterprise
    ]
)
CONTRACT_BY_SEGMENT = np.array(
    [
        [0.62, 0.26, 0.12],  # Consumer skews month-to-month
        [0.45, 0.35, 0.20],
        [0.20, 0.40, 0.40],  # Enterprise locks in longer terms
    ]
)

PAYMENT_METHODS = ["Credit card", "Bank transfer", "Electronic check", "Mailed check", "Direct debit"]
PAYMENT_METHOD_P = [0.42, 0.20, 0.18, 0.06, 0.14]

INCOME_LEVELS = ["Low", "Medium", "High", "Very High"]

CHURN_REASONS = [
    "Too expensive",
    "Poor customer service",
    "Found competitor",
    "Not using service",
    "Technical problems",
    "Missing features",
]

TICKET_CATEGORIES = [
    "Billing",
    "Technical issue",
    "Account access",
    "Feature request",
    "Service outage",
    "Cancellation enquiry",
]

# Disengagement dynamics
DISENGAGE_DECAY = 0.66      # monthly usage multiplier once a customer starts drifting away
RECOVERY_RATE = 0.20        # share of disengaging customers who come back without help

# Hazard model coefficients (the "true" data-generating process)
HAZARD = {
    "intercept": -4.05,
    "month_to_month": 1.15,
    "one_year": 0.45,
    "ticket_pressure": 0.62,
    "usage_decline": 1.05,
    "value_gap": 0.38,
    "late_payments": 0.60,
    "satisfaction": -0.50,
    "tenure": -0.026,
    "price_pressure": 0.55,
    "low_feature_use": 0.30,
    "new_customer": 0.55,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


# --------------------------------------------------------------------------------------
# Dimension builders
# --------------------------------------------------------------------------------------
def build_plans() -> pd.DataFrame:
    rows = []
    pid = 1
    for t, (name, price, limit, feats) in enumerate(PLAN_TIERS):
        for c, (contract, mult) in enumerate(CONTRACTS):
            rows.append(
                {
                    "plan_id": f"PL{pid:02d}",
                    "plan_name": f"{name} {contract}",
                    "plan_tier": name,
                    "tier_rank": t + 1,
                    "monthly_price": round(price * mult, 2),
                    "contract_type": contract,
                    "data_limit": limit,
                    "data_limit_gb": DATA_LIMIT_GB[t],
                    "features": feats,
                    "_tier_idx": t,
                    "_contract_idx": c,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def build_date_dimension(months: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_id": months.strftime("%Y%m%d").astype(int),
            "date": months.strftime("%Y-%m-%d"),
            "month": months.month,
            "month_name": months.strftime("%B"),
            "quarter": months.quarter,
            "year": months.year,
            "month_index": range(len(months)),
        }
    )


# --------------------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------------------
def generate(n_customers: int = N_CUSTOMERS, n_months: int = N_MONTHS, seed: int = SEED,
             write: bool = True, verbose: bool = True,
             dirty: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    months = pd.date_range(WINDOW_START, periods=n_months, freq="MS")
    dim_date = build_date_dimension(months)
    plans = build_plans()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ---------------------------------------------------------------- customer profiles
    log(f"Generating {n_customers:,} customer profiles ...")
    seg_idx = rng.choice(len(SEGMENTS), size=n_customers, p=SEGMENT_P)

    tier_idx = np.empty(n_customers, dtype=int)
    contract_idx = np.empty(n_customers, dtype=int)
    for s in range(len(SEGMENTS)):
        m = seg_idx == s
        tier_idx[m] = rng.choice(6, size=m.sum(), p=TIER_BY_SEGMENT[s])
        contract_idx[m] = rng.choice(3, size=m.sum(), p=CONTRACT_BY_SEGMENT[s])

    plan_lookup = plans.set_index(["_tier_idx", "_contract_idx"])
    plan_id = plan_lookup.loc[list(zip(tier_idx, contract_idx)), "plan_id"].to_numpy()
    monthly_price = plan_lookup.loc[list(zip(tier_idx, contract_idx)), "monthly_price"].to_numpy()
    data_limit_gb = np.array(DATA_LIMIT_GB)[tier_idx]
    n_features_in_plan = np.array([len(f.split(";")) for _, _, _, f in PLAN_TIERS])[tier_idx]

    # Tenure at the *end* of the observation window, in whole months.
    max_tenure = (
        (months[-1].year - pd.Timestamp(EARLIEST_SIGNUP).year) * 12
        + months[-1].month - pd.Timestamp(EARLIEST_SIGNUP).month
    )
    # Mixture: an established base plus a cohort that joined during the window (growth).
    established = rng.random(n_customers) < 0.55
    tenure_end = rng.exponential(scale=24, size=n_customers) + established * rng.uniform(12, 30, n_customers)
    tenure_end = np.clip(tenure_end.astype(int), 0, max_tenure)
    tenure_end[seg_idx == 2] = np.clip(tenure_end[seg_idx == 2] + rng.integers(0, 14, (seg_idx == 2).sum()), 0, max_tenure)

    # Offset of the signup month relative to the first month of the window
    signup_offset = (n_months - 1) - tenure_end          # negative => joined before the window
    signup_month = np.array(
        [months[0] - pd.DateOffset(months=int(-o)) if o < 0 else months[int(o)] for o in signup_offset]
    )
    signup_date = pd.to_datetime(signup_month) + pd.to_timedelta(rng.integers(0, 28, n_customers), unit="D")

    # Demographics — Faker is used to build pools, then sampled (fast and realistic:
    # customers concentrate in a finite set of cities and occupations).
    first_names = [fake.first_name() for _ in range(1500)]
    last_names = [fake.last_name() for _ in range(1500)]
    cities = [(fake.city(), fake.state()) for _ in range(220)]
    occupations = sorted({fake.job() for _ in range(400)})

    city_choice = rng.integers(0, len(cities), n_customers)
    customer_name = np.array(
        [f"{first_names[i]} {last_names[j]}"
         for i, j in zip(rng.integers(0, 1500, n_customers), rng.integers(0, 1500, n_customers))]
    )
    gender = rng.choice(["Male", "Female", "Other"], n_customers, p=[0.48, 0.48, 0.04])
    age = np.clip(rng.normal(41, 13, n_customers), 18, 84).astype(int)

    income_score = rng.normal(0, 1, n_customers) + 0.55 * tier_idx / 2 + 0.4 * (seg_idx == 2)
    income_level = np.array(INCOME_LEVELS)[np.clip(np.digitize(income_score, [-0.6, 0.4, 1.4]), 0, 3)]

    customers = pd.DataFrame(
        {
            "customer_id": [f"CUS{i:06d}" for i in range(1, n_customers + 1)],
            "customer_name": customer_name,
            "gender": gender,
            "age": age,
            "city": [cities[i][0] for i in city_choice],
            "state": [cities[i][1] for i in city_choice],
            "signup_date": pd.to_datetime(signup_date).strftime("%Y-%m-%d"),
            "customer_segment": np.array(SEGMENTS)[seg_idx],
            "occupation": rng.choice(occupations, n_customers),
            "income_level": income_level,
            "plan_id": plan_id,
        }
    )

    # ---------------------------------------------------------------- latent behaviour
    frailty = rng.normal(0, 0.55, n_customers)                       # unobserved risk appetite
    engagement_base = np.exp(rng.normal(np.log(5 + 4.5 * tier_idx), 0.45))
    trend_rate = rng.normal(-0.004, 0.032, n_customers)              # monthly multiplicative drift
    tech_propensity = rng.gamma(2.0, 0.5, n_customers)
    price_sensitivity = np.clip(rng.beta(2, 3, n_customers) + 0.25 * (seg_idx == 0) - 0.15 * (seg_idx == 2), 0.02, 1.0)
    satisfaction_base = np.clip(rng.normal(4.15, 0.55, n_customers) - 0.4 * price_sensitivity, 1.0, 5.0)
    gb_per_hour = rng.uniform(0.6, 2.6, n_customers)
    late_propensity = np.clip(rng.beta(1.6, 12, n_customers)
                              + 0.05 * (income_level == "Low")
                              + 0.02 * (rng.random(n_customers) < 0.3), 0, 0.9)
    pay_method = rng.choice(PAYMENT_METHODS, n_customers, p=PAYMENT_METHOD_P)
    late_propensity += 0.06 * (pay_method == "Electronic check") + 0.05 * (pay_method == "Mailed check")

    # ---------------------------------------------------------------- monthly simulation
    log(f"Simulating {n_months} months of behaviour ...")
    shape = (n_customers, n_months)
    active = np.zeros(shape, dtype=bool)
    usage_h = np.zeros(shape, dtype=np.float32)
    logins = np.zeros(shape, dtype=np.int16)
    tickets = np.zeros(shape, dtype=np.int16)
    data_gb = np.zeros(shape, dtype=np.float32)
    feat_use = np.zeros(shape, dtype=np.int16)
    satisfaction = np.zeros(shape, dtype=np.float32)
    late_flag = np.zeros(shape, dtype=np.int8)
    discount_flag = np.zeros(shape, dtype=np.int8)
    pay_amt = np.zeros(shape, dtype=np.float32)

    churned = np.zeros(n_customers, dtype=bool)
    churn_month = np.full(n_customers, -1, dtype=int)
    reason_scores = np.zeros((n_customers, len(CHURN_REASONS)), dtype=np.float32)

    # Two-stage churn. Customers rarely cancel out of nowhere: they disengage first.
    # `disengaging` marks the run-up phase, during which usage decays, tickets rise and
    # satisfaction falls. At the end of the phase the customer either cancels or is
    # recovered (which is what produces genuine false positives for the model).
    disengaging = np.zeros(n_customers, dtype=bool)
    disengage_start = np.full(n_customers, -1, dtype=int)
    disengage_len = rng.integers(2, 6, n_customers)          # months of visible decline
    will_recover = rng.random(n_customers) < RECOVERY_RATE

    price_z = _zscore(monthly_price)
    is_m2m = (contract_idx == 0).astype(float)
    is_1yr = (contract_idx == 1).astype(float)

    # Random "incident periods": a burst of technical problems for some customers
    incident_start = rng.integers(0, n_months, n_customers)
    has_incident = rng.random(n_customers) < 0.18

    for t in range(n_months):
        tenure_t = t - signup_offset                      # months since signup at month t
        alive = (~churned) & (tenure_t >= 0)
        if not alive.any():
            continue
        active[:, t] = alive
        idx = np.flatnonzero(alive)

        # Months elapsed inside the disengagement phase (0 for healthy customers)
        phase = np.where(disengaging[idx], t - disengage_start[idx] + 1, 0)
        decay = DISENGAGE_DECAY ** phase

        season = 1 + 0.07 * np.sin(2 * np.pi * (months[t].month - 3) / 12) - 0.10 * (months[t].month == 12)
        drift = np.exp(trend_rate[idx] * np.maximum(t, 0))
        u = engagement_base[idx] * drift * season * decay * np.exp(rng.normal(0, 0.22, idx.size))
        u = np.maximum(u, 0.05)
        usage_h[idx, t] = u

        logins[idx, t] = rng.poisson(np.clip(u * 1.15, 0.2, 400))
        data_gb[idx, t] = u * gb_per_hour[idx] * np.exp(rng.normal(0, 0.18, idx.size))
        engaged = np.clip(u / (engagement_base[idx] + 1e-6), 0, 2)
        feat_use[idx, t] = rng.binomial(n_features_in_plan[idx], np.clip(0.25 + 0.30 * engaged, 0.02, 0.98))

        incident = has_incident[idx] & (np.abs(t - incident_start[idx]) <= 1)
        lam = 0.14 * tech_propensity[idx] * (1 + 0.35 * engaged) + 0.9 * incident + 0.30 * phase
        tickets[idx, t] = rng.poisson(np.clip(lam, 0.01, 20))

        # Billing
        overage = np.maximum(data_gb[idx, t] - data_limit_gb[idx], 0) * 0.35
        disc = (rng.random(idx.size) < 0.12).astype(np.int8)
        discount_flag[idx, t] = disc
        pay_amt[idx, t] = monthly_price[idx] * (1 - 0.15 * disc) + overage
        late_flag[idx, t] = (rng.random(idx.size) < np.clip(late_propensity[idx] + 0.05 * phase, 0, 0.95)).astype(np.int8)

        satisfaction[idx, t] = np.clip(
            satisfaction_base[idx]
            - 0.30 * tickets[idx, t]
            - 0.35 * late_flag[idx, t]
            + 0.45 * (drift - 1)
            - 0.35 * phase
            + rng.normal(0, 0.30, idx.size),
            1.0, 5.0,
        )

        # ---------------- churn hazard for this month -----------------
        # Risk is read over multi-month windows, not a single month. Chronic ticket
        # pressure and a sustained usage decline are what actually precede cancellation;
        # a one-off spike usually does not.
        lo3, hi3 = max(t - 2, 0), t + 1
        lo6, hi6 = max(t - 5, 0), t + 1
        lo9, hi9 = max(t - 8, 0), max(t - 3, 0)
        tick6 = tickets[idx, lo6:hi6].mean(axis=1)
        sat3 = satisfaction[idx, lo3:hi3].mean(axis=1)
        late3 = late_flag[idx, lo6:hi6].mean(axis=1)
        use3 = usage_h[idx, lo3:hi3].mean(axis=1)
        use_prev = usage_h[idx, lo9:hi9].mean(axis=1) if hi9 > lo9 else use3
        decline = np.clip((use_prev - use3) / (use_prev + 1e-3), -1, 1)
        feat_ratio = feat_use[idx, t] / np.maximum(n_features_in_plan[idx], 1)
        # Value gap: paying a premium price for very little usage is a persistent,
        # observable reason to leave.
        value_gap = np.log1p(monthly_price[idx] / (use3 + 1.0))
        tick3 = tick6

        z = (
            HAZARD["intercept"]
            + HAZARD["month_to_month"] * is_m2m[idx]
            + HAZARD["one_year"] * is_1yr[idx]
            + HAZARD["ticket_pressure"] * np.clip(tick6, 0, 6)
            + HAZARD["usage_decline"] * decline
            + HAZARD["late_payments"] * late3 * 2
            + HAZARD["satisfaction"] * (sat3 - 3.9)
            + HAZARD["tenure"] * np.minimum(tenure_t[idx], 48)
            + HAZARD["price_pressure"] * price_sensitivity[idx] * price_z[idx]
            + HAZARD["low_feature_use"] * (feat_ratio < 0.25)
            + HAZARD["value_gap"] * (value_gap - 2.2)
            + HAZARD["new_customer"] * (tenure_t[idx] < 3)
            + frailty[idx]
        )
        # Stage 1: healthy customers may enter the disengagement phase this month.
        p = _sigmoid(z)
        enters = (~disengaging[idx]) & (rng.random(idx.size) < p)
        leaving = idx[enters]

        if leaving.size:
            disengaging[leaving] = True
            disengage_start[leaving] = t
            sel = enters
            reason_scores[leaving, 0] = 1.5 * price_sensitivity[leaving] * price_z[leaving] + 0.4   # too expensive
            reason_scores[leaving, 1] = 0.9 * np.clip(tick3[sel], 0, 6) - 0.7 * (sat3[sel] - 3.5)   # poor service
            reason_scores[leaving, 2] = 0.6 + 0.5 * is_m2m[leaving] + 0.3 * price_z[leaving]        # competitor
            reason_scores[leaving, 3] = 1.6 * decline[sel] + 0.9 * (feat_ratio[sel] < 0.25)         # not using
            reason_scores[leaving, 4] = 1.1 * np.clip(tick3[sel], 0, 6) * has_incident[leaving]     # technical
            reason_scores[leaving, 5] = 0.5 + 0.8 * (tier_idx[leaving] <= 1) * (usage_h[leaving, t] > engagement_base[leaving])

        # Stage 2: phases that mature this month resolve into a cancellation, unless the
        # customer recovers on their own.
        maturing = idx[disengaging[idx] & (t >= disengage_start[idx] + disengage_len[idx])]
        if maturing.size:
            leave = maturing[~will_recover[maturing]]
            churned[leave] = True
            churn_month[leave] = t
            recovered = maturing[will_recover[maturing]]
            disengaging[recovered] = False
            disengage_start[recovered] = -1
            will_recover[recovered] = rng.random(recovered.size) < RECOVERY_RATE

    log("Assembling fact tables ...")

    # ---------------------------------------------------------------- churn status
    reason = np.full(n_customers, "", dtype=object)
    ch = np.flatnonzero(churned)
    gumbel = rng.gumbel(0, 0.8, (ch.size, len(CHURN_REASONS)))
    reason[ch] = np.array(CHURN_REASONS)[(reason_scores[ch] + gumbel).argmax(axis=1)]

    churn_date = np.full(n_customers, "", dtype=object)
    for i in ch:
        m = months[churn_month[i]]
        churn_date[i] = (m + pd.Timedelta(days=int(rng.integers(0, 28)))).strftime("%Y-%m-%d")

    status = pd.DataFrame(
        {
            "customer_id": customers["customer_id"],
            "churn_flag": churned.astype(int),
            "churn_date": churn_date,
            "cancellation_reason": reason,
        }
    )

    # ---------------------------------------------------------------- long fact tables
    cust_ix, month_ix = np.nonzero(active)
    date_ids = dim_date["date_id"].to_numpy()[month_ix]
    cid = customers["customer_id"].to_numpy()[cust_ix]

    usage = pd.DataFrame(
        {
            "customer_id": cid,
            "date_id": date_ids,
            "login_frequency": logins[cust_ix, month_ix],
            "support_tickets": tickets[cust_ix, month_ix],
            "usage_hours": np.round(usage_h[cust_ix, month_ix], 2),
            "data_consumption": np.round(data_gb[cust_ix, month_ix], 2),
            "feature_usage": feat_use[cust_ix, month_ix],
            "satisfaction_score": np.round(satisfaction[cust_ix, month_ix], 2),
        }
    )

    payments = pd.DataFrame(
        {
            "customer_id": cid,
            "date_id": date_ids,
            "payment_amount": np.round(pay_amt[cust_ix, month_ix], 2),
            "payment_method": pay_method[cust_ix],
            "late_payment": late_flag[cust_ix, month_ix],
            "discount_used": discount_flag[cust_ix, month_ix],
        }
    )

    # Ticket-level support events (one row per ticket)
    tk = tickets[cust_ix, month_ix]
    rep = tk > 0
    support = pd.DataFrame(
        {
            "customer_id": np.repeat(cid[rep], tk[rep]),
            "date_id": np.repeat(date_ids[rep], tk[rep]),
        }
    )
    n_tickets = len(support)
    support.insert(0, "ticket_id", [f"TK{i:08d}" for i in range(1, n_tickets + 1)])
    support["category"] = rng.choice(TICKET_CATEGORIES, n_tickets, p=[0.22, 0.30, 0.14, 0.12, 0.10, 0.12])
    support["priority"] = rng.choice(["Low", "Medium", "High", "Critical"], n_tickets, p=[0.34, 0.40, 0.20, 0.06])
    support["resolution_hours"] = np.round(rng.gamma(2.0, 6.0, n_tickets), 1)
    support["first_contact_resolved"] = (rng.random(n_tickets) < 0.58).astype(int)

    tables = {
        "customers": customers,
        "plans": plans.drop(columns=["_tier_idx", "_contract_idx"]),
        "dim_date": dim_date,
        "usage": usage,
        "payments": payments,
        "support": support,
        "status": status,
    }

    # The raw layer carries production-style defects whether or not it is written to disk,
    # so tests and notebooks exercise the same extract the pipeline sees.
    if dirty:
        tables = _inject_realistic_dirt(tables, rng)

    if write:
        log("Writing raw CSVs ...")
        for name, df in tables.items():
            df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        log(
            f"Done. customers={len(customers):,} usage_rows={len(usage):,} "
            f"payments={len(payments):,} tickets={len(support):,} "
            f"churn_rate={churned.mean():.1%}"
        )
    return tables


# --------------------------------------------------------------------------------------
# Dirt injection - the raw layer should look like an export from a production system
# --------------------------------------------------------------------------------------
def _inject_realistic_dirt(tables: dict[str, pd.DataFrame], rng) -> dict[str, pd.DataFrame]:
    """Introduce the defects a real extract would contain, so the cleaning step is real."""
    cust = tables["customers"].copy()
    cust["age"] = cust["age"].astype("float64")   # allow nulls / bad values below
    n = len(cust)

    # 1. Inconsistent categorical spellings
    gmap = {"Male": ["Male", "male", "M", "MALE"], "Female": ["Female", "female", "F", "FEMALE"], "Other": ["Other", "other"]}
    cust["gender"] = [rng.choice(gmap[g]) for g in cust["gender"]]

    # 2. Stray whitespace / casing in free-text location
    noisy = rng.random(n) < 0.08
    cust.loc[noisy, "city"] = "  " + cust.loc[noisy, "city"].str.upper() + " "

    # 3. Missing values
    for col, frac in [("age", 0.025), ("income_level", 0.035), ("occupation", 0.02)]:
        miss = rng.random(n) < frac
        cust.loc[miss, col] = np.nan

    # 4. Impossible ages (data-entry errors)
    bad_age = rng.choice(n, size=60, replace=False)
    cust.loc[bad_age, "age"] = rng.choice([0, 1, 199, 250], size=60)

    # 5. Duplicated customer rows (a classic double-load)
    dupes = cust.sample(600, random_state=1)
    cust = pd.concat([cust, dupes], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)
    tables["customers"] = cust

    # 6. Payment defects: inconsistent method labels, negative amounts, missing values
    pay = tables["payments"].copy()
    pay["payment_amount"] = pay["payment_amount"].astype("float64").round(2)
    pmap = {
        "Credit card": ["Credit card", "credit card", "CC", "Credit Card"],
        "Bank transfer": ["Bank transfer", "bank transfer", "BANK TRANSFER"],
        "Electronic check": ["Electronic check", "E-check", "electronic check"],
        "Mailed check": ["Mailed check", "mailed check"],
        "Direct debit": ["Direct debit", "direct debit", "DD"],
    }
    variant = {k: rng.choice(v, size=len(v), replace=False) for k, v in pmap.items()}
    pay["payment_method"] = [rng.choice(variant[m]) for m in pay["payment_method"]]
    neg = rng.choice(len(pay), size=int(0.001 * len(pay)), replace=False)
    pay.loc[neg, "payment_amount"] *= -1                      # refunds mis-booked as payments
    miss = rng.random(len(pay)) < 0.004
    pay.loc[miss, "payment_amount"] = np.nan
    tables["payments"] = pay

    # 7. Usage defects: sensor spikes and missing readings
    use = tables["usage"].copy()
    use["usage_hours"] = use["usage_hours"].astype("float64").round(2)
    spike = rng.choice(len(use), size=int(0.0006 * len(use)), replace=False)
    use.loc[spike, "usage_hours"] *= rng.uniform(40, 120, len(spike))
    miss = rng.random(len(use)) < 0.006
    use.loc[miss, "usage_hours"] = np.nan
    tables["usage"] = use

    return tables


if __name__ == "__main__":
    generate()
