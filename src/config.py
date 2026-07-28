"""
Central configuration for the Customer Churn Prediction & Retention Intelligence Platform.

Every path, date boundary and modelling constant lives here so that notebooks,
scripts and the dashboard all read from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATABASE_DIR = PROJECT_ROOT / "database"

for _p in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Simulation window
# --------------------------------------------------------------------------------------
SEED = 42
N_CUSTOMERS = 50_000
N_MONTHS = 24
WINDOW_START = "2024-08-01"          # first month of observed history
EARLIEST_SIGNUP = "2019-01-01"       # customers can have joined long before the window

# --------------------------------------------------------------------------------------
# Modelling design
# --------------------------------------------------------------------------------------
# Customers are scored at the end of the observation window. Everything the model sees is
# built from months <= CUTOFF; the label is "did this customer cancel during the following
# HORIZON_MONTHS". Keeping a strict time gap between features and label is what stops
# target leakage.
OBSERVATION_MONTHS = 21              # months 1-21 -> features
HORIZON_MONTHS = 3                   # months 22-24 -> label

TEST_SIZE = 0.20
VALID_SIZE = 0.20                    # carved out of the training portion
RANDOM_STATE = 42

# Risk banding used across segmentation, CLV and the dashboard
RISK_BANDS = [
    ("Low", 0.00, 0.30),
    ("Medium", 0.30, 0.70),
    ("High", 0.70, 1.01),
]

# Business assumptions used by the CLV / revenue-at-risk calculations
GROSS_MARGIN = 0.72                  # typical SaaS gross margin
DISCOUNT_RATE_MONTHLY = 0.01         # ~12.7% annual, used to discount future cash flows
RETENTION_OFFER_COST = 0.20          # a 20% discount is the standard save offer
ASSUMED_SAVE_RATE = 0.30             # share of contacted high-risk customers actually saved

# Database (ETL load target). The pipeline writes to SQLite by default so the project runs
# with zero setup; set CHURN_DB_URL to a PostgreSQL URI to load the real warehouse.
DEFAULT_DB_URL = f"sqlite:///{(DATABASE_DIR / 'churn_warehouse.db').as_posix()}"
POSTGRES_EXAMPLE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/churn_dw"
