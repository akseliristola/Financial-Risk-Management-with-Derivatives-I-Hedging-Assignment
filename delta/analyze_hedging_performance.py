import pandas as pd

def main():
    raw_path = "delta_hedging_performance_raw.csv"
    summary_path = "delta_hedging_performance_summary.csv"

    # Load raw performance data
    df = pd.read_csv(raw_path)

    # Sanity check: ensure the key columns exist
    required_cols = [
        "maturity",
        "interval_days",
        "strike_type",
        "brokerage_fee",
        "mse",
        "iterations",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # Group by C1 strike type, hedging interval, and brokerage fee
    group_cols = ["strike_type", "interval_days", "brokerage_fee"]

    # Aggregate hedging error statistics
    summary = (
        df.groupby(group_cols)
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            min_mse=("mse", "min"),
            max_mse=("mse", "max"),
        )
        .reset_index()
    )

    # Save summary to CSV
    summary.to_csv(summary_path, index=False)

    # Print a human-readable version to the console
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
    print("Hedging performance summary (by strike_type, interval_days, brokerage_fee):")
    print(summary)

if __name__ == "__main__":
    main()
