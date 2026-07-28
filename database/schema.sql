-- =====================================================================================
-- Customer Churn Prediction & Retention Intelligence Platform
-- Analytical warehouse - star schema (PostgreSQL)
--
-- Grain
--   dim_customer          one row per customer
--   dim_plan              one row per plan tier x contract term
--   dim_date              one row per month in the observation window
--   fact_customer_usage   one row per customer per month  (engagement)
--   fact_payments         one row per customer per month  (billing)
--   fact_support_tickets  one row per support ticket      (event grain)
--   fact_customer_status  one row per customer            (churn outcome)
--   fact_churn_scores     one row per customer per scoring run (model output)
--
-- Run with:  psql -d churn_dw -f database/schema.sql
-- =====================================================================================

DROP TABLE IF EXISTS fact_churn_scores    CASCADE;
DROP TABLE IF EXISTS fact_customer_status CASCADE;
DROP TABLE IF EXISTS fact_support_tickets CASCADE;
DROP TABLE IF EXISTS fact_payments        CASCADE;
DROP TABLE IF EXISTS fact_customer_usage  CASCADE;
DROP TABLE IF EXISTS dim_date             CASCADE;
DROP TABLE IF EXISTS dim_plan             CASCADE;
DROP TABLE IF EXISTS dim_customer         CASCADE;

-- -------------------------------------------------------------------------------------
-- Dimensions
-- -------------------------------------------------------------------------------------
CREATE TABLE dim_plan (
    plan_id         VARCHAR(8)   PRIMARY KEY,
    plan_name       VARCHAR(60)  NOT NULL,
    plan_tier       VARCHAR(20)  NOT NULL,
    tier_rank       SMALLINT     NOT NULL,
    monthly_price   NUMERIC(8,2) NOT NULL CHECK (monthly_price >= 0),
    contract_type   VARCHAR(20)  NOT NULL,
    data_limit      VARCHAR(20),
    data_limit_gb   INTEGER,
    features        TEXT
);

CREATE TABLE dim_customer (
    customer_id       VARCHAR(12) PRIMARY KEY,
    customer_name     VARCHAR(80),
    gender            VARCHAR(10),
    age               SMALLINT CHECK (age BETWEEN 16 AND 100),
    location          VARCHAR(120),          -- "City, State"
    city              VARCHAR(60),
    state             VARCHAR(40),
    region            VARCHAR(20),
    signup_date       DATE NOT NULL,
    customer_segment  VARCHAR(20),
    occupation        VARCHAR(120),
    income_level      VARCHAR(20),
    plan_id           VARCHAR(8) REFERENCES dim_plan (plan_id)
);

CREATE TABLE dim_date (
    date_id      INTEGER PRIMARY KEY,        -- YYYYMMDD of the first day of the month
    date         DATE        NOT NULL,
    month        SMALLINT    NOT NULL,
    month_name   VARCHAR(12) NOT NULL,
    quarter      SMALLINT    NOT NULL,
    year         SMALLINT    NOT NULL,
    month_index  SMALLINT    NOT NULL        -- 0-based position in the window
);

-- -------------------------------------------------------------------------------------
-- Facts
-- -------------------------------------------------------------------------------------
CREATE TABLE fact_customer_usage (
    customer_id        VARCHAR(12) NOT NULL REFERENCES dim_customer (customer_id),
    date_id            INTEGER     NOT NULL REFERENCES dim_date (date_id),
    login_frequency    INTEGER,
    support_tickets    SMALLINT,
    usage_hours        NUMERIC(10,2),
    data_consumption   NUMERIC(12,2),
    feature_usage      SMALLINT,
    satisfaction_score NUMERIC(4,2),
    PRIMARY KEY (customer_id, date_id)
);

CREATE TABLE fact_payments (
    customer_id     VARCHAR(12) NOT NULL REFERENCES dim_customer (customer_id),
    date_id         INTEGER     NOT NULL REFERENCES dim_date (date_id),
    payment_amount  NUMERIC(10,2),
    payment_method  VARCHAR(30),
    late_payment    SMALLINT,
    discount_used   SMALLINT,
    is_refund       SMALLINT,
    PRIMARY KEY (customer_id, date_id)
);

CREATE TABLE fact_support_tickets (
    ticket_id              VARCHAR(12) PRIMARY KEY,
    customer_id            VARCHAR(12) NOT NULL REFERENCES dim_customer (customer_id),
    date_id                INTEGER     NOT NULL REFERENCES dim_date (date_id),
    category               VARCHAR(40),
    priority               VARCHAR(15),
    resolution_hours       NUMERIC(8,1),
    first_contact_resolved SMALLINT
);

CREATE TABLE fact_customer_status (
    customer_id         VARCHAR(12) PRIMARY KEY REFERENCES dim_customer (customer_id),
    churn_flag          SMALLINT NOT NULL,
    churn_date          DATE,
    cancellation_reason VARCHAR(60)
);

CREATE TABLE fact_churn_scores (
    customer_id               VARCHAR(12) NOT NULL REFERENCES dim_customer (customer_id),
    scoring_date              DATE        NOT NULL,
    churn_probability         NUMERIC(6,5) NOT NULL,
    risk_band                 VARCHAR(10)  NOT NULL,
    expected_remaining_months NUMERIC(6,1),
    clv                       NUMERIC(12,2),
    arr_at_risk               NUMERIC(12,2),
    recommended_action        VARCHAR(120),
    action_owner              VARCHAR(40),
    action_expected_value     NUMERIC(12,2),
    PRIMARY KEY (customer_id, scoring_date)
);

-- -------------------------------------------------------------------------------------
-- Indexes for the analytical access patterns
-- -------------------------------------------------------------------------------------
CREATE INDEX idx_usage_date        ON fact_customer_usage (date_id);
CREATE INDEX idx_payments_date     ON fact_payments (date_id);
CREATE INDEX idx_tickets_customer  ON fact_support_tickets (customer_id);
CREATE INDEX idx_tickets_date      ON fact_support_tickets (date_id);
CREATE INDEX idx_status_churn      ON fact_customer_status (churn_flag);
CREATE INDEX idx_customer_plan     ON dim_customer (plan_id);
CREATE INDEX idx_customer_segment  ON dim_customer (customer_segment);
CREATE INDEX idx_scores_band       ON fact_churn_scores (risk_band);

-- -------------------------------------------------------------------------------------
-- Reporting views consumed by the dashboard and BI layer
-- -------------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_customer_360 AS
SELECT
    c.customer_id,
    c.customer_name,
    c.customer_segment,
    c.region,
    c.signup_date,
    p.plan_name,
    p.plan_tier,
    p.contract_type,
    p.monthly_price,
    s.churn_flag,
    s.churn_date,
    s.cancellation_reason,
    u.months_active,
    u.avg_usage_hours,
    u.total_tickets,
    u.avg_satisfaction,
    b.avg_monthly_spend,
    b.total_revenue,
    b.late_payments
FROM dim_customer c
JOIN dim_plan p              ON p.plan_id = c.plan_id
LEFT JOIN fact_customer_status s ON s.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id,
           COUNT(*)                 AS months_active,
           AVG(usage_hours)         AS avg_usage_hours,
           SUM(support_tickets)     AS total_tickets,
           AVG(satisfaction_score)  AS avg_satisfaction
    FROM fact_customer_usage GROUP BY customer_id
) u ON u.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id,
           AVG(payment_amount) AS avg_monthly_spend,
           SUM(payment_amount) AS total_revenue,
           SUM(late_payment)   AS late_payments
    FROM fact_payments GROUP BY customer_id
) b ON b.customer_id = c.customer_id;

CREATE OR REPLACE VIEW vw_monthly_churn AS
SELECT
    d.year,
    d.month,
    d.date                                   AS month_start,
    COUNT(DISTINCT u.customer_id)            AS active_customers,
    COUNT(DISTINCT s.customer_id)            AS churned_customers,
    ROUND(COUNT(DISTINCT s.customer_id)::NUMERIC
          / NULLIF(COUNT(DISTINCT u.customer_id), 0), 4) AS churn_rate
FROM dim_date d
LEFT JOIN fact_customer_usage u  ON u.date_id = d.date_id
LEFT JOIN fact_customer_status s ON DATE_TRUNC('month', s.churn_date) = d.date
GROUP BY d.year, d.month, d.date
ORDER BY d.date;

CREATE OR REPLACE VIEW vw_revenue_at_risk AS
SELECT
    f.risk_band,
    COUNT(*)                                   AS customers,
    ROUND(AVG(f.churn_probability), 4)         AS avg_probability,
    ROUND(SUM(f.arr_at_risk), 2)               AS arr_at_risk,
    ROUND(SUM(f.clv), 2)                       AS total_clv,
    ROUND(SUM(f.action_expected_value), 2)     AS expected_value_of_action
FROM fact_churn_scores f
GROUP BY f.risk_band;
