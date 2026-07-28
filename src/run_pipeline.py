"""
Run the whole platform end to end.

    python src/run_pipeline.py              # full rebuild
    python src/run_pipeline.py --skip-data  # reuse the existing raw extracts

Stages:
    1. generate   synthetic raw extracts (50k customers x 24 months)
    2. transform  clean, build the customer-month panel, engineer features
    3. load       push the star schema into SQLite (or PostgreSQL via CHURN_DB_URL)
    4. train      fit and compare four models, choose an operating threshold
    5. explain    SHAP global and per-customer drivers
    6. score      probabilities, risk bands, CLV, retention playbook
    7. report     every figure used in the README and the dashboard
"""

from __future__ import annotations

import argparse
import time


def _stage(n: int, total: int, name: str) -> float:
    print(f"\n[{n}/{total}] {name}\n" + "-" * 72, flush=True)
    return time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer churn pipeline")
    parser.add_argument("--skip-data", action="store_true", help="reuse existing data/raw CSVs")
    parser.add_argument("--skip-db", action="store_true", help="skip the warehouse load")
    parser.add_argument("--customers", type=int, default=None, help="override customer count")
    args = parser.parse_args()

    import data_generator
    import etl_load
    import explain
    import model
    import predictions
    import preprocessing
    import visualisation

    total, started = 7, time.time()

    if args.skip_data:
        print("[1/7] Generate — skipped, reusing data/raw")
    else:
        t = _stage(1, total, "Generate synthetic extracts")
        kwargs = {"n_customers": args.customers} if args.customers else {}
        data_generator.generate(**kwargs)
        print(f"    done in {time.time() - t:.0f}s")

    t = _stage(2, total, "Transform — clean and engineer features")
    preprocessing.main()
    print(f"    done in {time.time() - t:.0f}s")

    if args.skip_db:
        print("\n[3/7] Load — skipped")
    else:
        t = _stage(3, total, "Load — build the analytical warehouse")
        engine = etl_load.get_engine()
        etl_load.load_all(engine)
        print(f"    done in {time.time() - t:.0f}s")

    t = _stage(4, total, "Train — fit and compare models")
    model.main()
    print(f"    done in {time.time() - t:.0f}s")

    t = _stage(5, total, "Explain — SHAP drivers")
    explain.main()
    print(f"    done in {time.time() - t:.0f}s")

    t = _stage(6, total, "Score — risk bands, CLV and retention plays")
    predictions.main()
    print(f"    done in {time.time() - t:.0f}s")

    t = _stage(7, total, "Report — figures and summary tables")
    visualisation.main()
    print(f"    done in {time.time() - t:.0f}s")

    if not args.skip_db:
        engine = etl_load.get_engine()
        etl_load.load_all(engine)     # refresh fact_churn_scores with the new run

    print(f"\nPipeline complete in {(time.time() - started)/60:.1f} minutes.")
    print("Launch the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
