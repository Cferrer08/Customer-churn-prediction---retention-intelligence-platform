-- =====================================================================================
-- Analytical queries against the churn warehouse.
--
-- These answer the questions the business actually asked, in SQL rather than pandas,
-- and are the source for several of the numbers quoted in the README.
-- Written for PostgreSQL; the SQLite build of the warehouse runs all of them except
-- where noted.
-- =====================================================================================

-- -------------------------------------------------------------------------------------
-- 1. Headline: how many customers, how many left, what did it cost?
-- -------------------------------------------------------------------------------------
SELECT
    COUNT(*)                                                        AS total_customers,
    SUM(churn_flag)                                                 AS churned_customers,
    ROUND(AVG(churn_flag::NUMERIC), 4)                              AS churn_rate,
    ROUND(SUM(CASE WHEN churn_flag = 1 THEN monthly_price ELSE 0 END) * 12, 2)
                                                                    AS annualised_revenue_lost
FROM fact_customer_status s
JOIN dim_customer c USING (customer_id)
JOIN dim_plan     p USING (plan_id);


-- -------------------------------------------------------------------------------------
-- 2. Which plans have the highest churn, and is it worth caring about?
--    Ranked by revenue impact rather than by rate alone - a 40% churn rate on a
--    $19 plan matters less than 12% on a $250 plan.
-- -------------------------------------------------------------------------------------
SELECT
    p.plan_tier,
    p.contract_type,
    COUNT(*)                                        AS customers,
    SUM(s.churn_flag)                               AS churned,
    ROUND(AVG(s.churn_flag::NUMERIC), 4)            AS churn_rate,
    ROUND(SUM(s.churn_flag * p.monthly_price) * 12, 2) AS annual_revenue_lost
FROM dim_customer c
JOIN dim_plan p              USING (plan_id)
JOIN fact_customer_status s  USING (customer_id)
GROUP BY p.plan_tier, p.contract_type
HAVING COUNT(*) > 100
ORDER BY annual_revenue_lost DESC;


-- -------------------------------------------------------------------------------------
-- 3. Does contract type affect churn? Controlled for plan tier, so the answer is not
--    just "cheap customers churn".
-- -------------------------------------------------------------------------------------
SELECT
    p.tier_rank,
    p.plan_tier,
    ROUND(AVG(CASE WHEN p.contract_type = 'Month-to-month' THEN s.churn_flag::NUMERIC END), 4) AS m2m_churn,
    ROUND(AVG(CASE WHEN p.contract_type = 'One year'       THEN s.churn_flag::NUMERIC END), 4) AS one_year_churn,
    ROUND(AVG(CASE WHEN p.contract_type = 'Two year'       THEN s.churn_flag::NUMERIC END), 4) AS two_year_churn
FROM dim_customer c
JOIN dim_plan p             USING (plan_id)
JOIN fact_customer_status s USING (customer_id)
GROUP BY p.tier_rank, p.plan_tier
ORDER BY p.tier_rank;


-- -------------------------------------------------------------------------------------
-- 4. Signup-cohort retention curve: of everyone who joined in month X,
--    what share is still active N months later?
-- -------------------------------------------------------------------------------------
WITH cohorts AS (
    SELECT
        c.customer_id,
        DATE_TRUNC('month', c.signup_date)                AS cohort_month,
        COALESCE(
            DATE_PART('year',  AGE(s.churn_date, c.signup_date)) * 12 +
            DATE_PART('month', AGE(s.churn_date, c.signup_date)),
            999
        )                                                  AS months_to_churn
    FROM dim_customer c
    JOIN fact_customer_status s USING (customer_id)
)
SELECT
    cohort_month,
    COUNT(*)                                                             AS cohort_size,
    ROUND(AVG(CASE WHEN months_to_churn >  3 THEN 1.0 ELSE 0 END), 3)    AS retained_3m,
    ROUND(AVG(CASE WHEN months_to_churn >  6 THEN 1.0 ELSE 0 END), 3)    AS retained_6m,
    ROUND(AVG(CASE WHEN months_to_churn > 12 THEN 1.0 ELSE 0 END), 3)    AS retained_12m
FROM cohorts
GROUP BY cohort_month
HAVING COUNT(*) > 50
ORDER BY cohort_month;


-- -------------------------------------------------------------------------------------
-- 5. The core behavioural finding: usage collapses before cancellation.
--    Window functions align each customer's last six months against their churn date.
-- -------------------------------------------------------------------------------------
WITH churn_months AS (
    SELECT customer_id, DATE_TRUNC('month', churn_date) AS churn_month
    FROM fact_customer_status
    WHERE churn_flag = 1
),
aligned AS (
    SELECT
        u.customer_id,
        (DATE_PART('year',  AGE(cm.churn_month, d.date)) * 12 +
         DATE_PART('month', AGE(cm.churn_month, d.date)))::INT AS months_before_churn,
        u.usage_hours,
        u.support_tickets,
        u.satisfaction_score
    FROM fact_customer_usage u
    JOIN dim_date    d  ON d.date_id = u.date_id
    JOIN churn_months cm ON cm.customer_id = u.customer_id
)
SELECT
    months_before_churn,
    COUNT(*)                              AS observations,
    ROUND(AVG(usage_hours), 2)            AS avg_usage_hours,
    ROUND(AVG(support_tickets), 3)        AS avg_tickets,
    ROUND(AVG(satisfaction_score), 2)     AS avg_satisfaction
FROM aligned
WHERE months_before_churn BETWEEN 0 AND 6
GROUP BY months_before_churn
ORDER BY months_before_churn DESC;


-- -------------------------------------------------------------------------------------
-- 6. Does usage intensity predict churn? Quintiles of average monthly usage.
-- -------------------------------------------------------------------------------------
WITH usage_per_customer AS (
    SELECT customer_id, AVG(usage_hours) AS avg_usage
    FROM fact_customer_usage
    GROUP BY customer_id
),
banded AS (
    SELECT
        u.customer_id,
        NTILE(5) OVER (ORDER BY u.avg_usage) AS usage_quintile,
        s.churn_flag
    FROM usage_per_customer u
    JOIN fact_customer_status s USING (customer_id)
)
SELECT
    usage_quintile,
    COUNT(*)                              AS customers,
    ROUND(AVG(churn_flag::NUMERIC), 4)    AS churn_rate
FROM banded
GROUP BY usage_quintile
ORDER BY usage_quintile;


-- -------------------------------------------------------------------------------------
-- 7. Support pressure vs churn, plus the first-contact-resolution angle.
-- -------------------------------------------------------------------------------------
WITH ticket_load AS (
    SELECT
        customer_id,
        COUNT(*)                                    AS tickets,
        AVG(first_contact_resolved::NUMERIC)        AS fcr_rate,
        AVG(resolution_hours)                       AS avg_resolution_hours
    FROM fact_support_tickets
    GROUP BY customer_id
)
SELECT
    CASE
        WHEN COALESCE(t.tickets, 0) = 0 THEN '0 tickets'
        WHEN t.tickets BETWEEN 1 AND 2  THEN '1-2 tickets'
        WHEN t.tickets BETWEEN 3 AND 5  THEN '3-5 tickets'
        ELSE '6+ tickets'
    END                                             AS ticket_band,
    COUNT(*)                                        AS customers,
    ROUND(AVG(s.churn_flag::NUMERIC), 4)            AS churn_rate,
    ROUND(AVG(t.fcr_rate), 3)                       AS avg_first_contact_resolution,
    ROUND(AVG(t.avg_resolution_hours), 1)           AS avg_resolution_hours
FROM fact_customer_status s
LEFT JOIN ticket_load t USING (customer_id)
GROUP BY ticket_band
ORDER BY churn_rate DESC;


-- -------------------------------------------------------------------------------------
-- 8. Why customers said they left, weighted by the revenue that walked out.
-- -------------------------------------------------------------------------------------
SELECT
    s.cancellation_reason,
    COUNT(*)                                       AS customers,
    ROUND(AVG(p.monthly_price), 2)                 AS avg_monthly_price,
    ROUND(SUM(p.monthly_price) * 12, 2)            AS annualised_revenue_lost,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_churn
FROM fact_customer_status s
JOIN dim_customer c USING (customer_id)
JOIN dim_plan     p USING (plan_id)
WHERE s.churn_flag = 1
GROUP BY s.cancellation_reason
ORDER BY annualised_revenue_lost DESC;


-- -------------------------------------------------------------------------------------
-- 9. Current book of business: where is the revenue at risk concentrated?
-- -------------------------------------------------------------------------------------
SELECT
    c.customer_segment,
    f.risk_band,
    COUNT(*)                                  AS customers,
    ROUND(SUM(f.arr_at_risk), 2)              AS arr_at_risk,
    ROUND(SUM(f.clv), 2)                      AS portfolio_clv,
    ROUND(AVG(f.expected_remaining_months), 1) AS avg_expected_months_remaining
FROM fact_churn_scores f
JOIN dim_customer c USING (customer_id)
GROUP BY c.customer_segment, f.risk_band
ORDER BY arr_at_risk DESC;


-- -------------------------------------------------------------------------------------
-- 10. The call list: highest-value customers the retention desk should work today.
-- -------------------------------------------------------------------------------------
SELECT
    f.customer_id,
    c.customer_name,
    c.customer_segment,
    p.plan_name,
    p.contract_type,
    ROUND(f.churn_probability, 3)      AS churn_probability,
    ROUND(f.arr_at_risk, 2)            AS arr_at_risk,
    f.recommended_action,
    f.action_owner,
    ROUND(f.action_expected_value, 2)  AS expected_value
FROM fact_churn_scores f
JOIN dim_customer c USING (customer_id)
JOIN dim_plan     p USING (plan_id)
WHERE f.risk_band = 'High'
ORDER BY f.action_expected_value DESC
LIMIT 100;
