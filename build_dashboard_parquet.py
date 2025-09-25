#!/usr/bin/env python3
"""
build_dashboard_parquet.py
---------------------------------
Reads the unified forecasts CSV and writes the Streamlit-friendly parquet
to Dashboard/data/forecasts.parquet.

This is safe to run locally anytime. It will create the Dashboard/data/
folder if it doesn't exist.

Notes for non-coders:
- By default, Spagbot logs all forecasts to forecast_logs/forecasts.csv (see io_logs.py).
- If you've changed FORECASTS_CSV_PATH in your environment, we'll honor that.
"""

import os
import sys

# Pandas is used to read/write CSV/Parquet.
# Parquet writing needs pyarrow installed (pip install pyarrow).
try:
    import pandas as pd
except Exception as e:
    sys.exit(
        "❌ pandas is not installed. In a terminal run:\n"
        "   pip install pandas pyarrow\n"
        f"Details: {e!r}"
    )

def main() -> None:
    # Where the unified CSV usually lives (see io_logs.get_log_paths)
    csv_path = os.getenv("FORECASTS_CSV_PATH", "forecast_logs/forecasts.csv")
    if not os.path.exists(csv_path):
        # Accept a fallback at repo root if you’ve exported it there
        alt = "forecasts.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            sys.exit(
                f"❌ Could not find forecasts CSV at:\n"
                f"   {os.path.abspath(csv_path)}\n"
                f"   (or fallback {os.path.abspath(alt)})\n"
                "Tip: run the bot once so forecasts.csv exists, or copy the CSV in."
            )

    print(f"ℹ️  Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    out_dir = os.path.join("Dashboard", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_pq = os.path.join(out_dir, "forecasts.parquet")

    print(f"ℹ️  Writing {out_pq} ({len(df):,} rows) ...")
    # Requires pyarrow installed
    df.to_parquet(out_pq, index=False)
    print("✅ Done.")

if __name__ == "__main__":
    main()
