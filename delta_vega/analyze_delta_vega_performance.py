import pandas as pd
import os
import sys

# Define constants for file names
# Assuming delta_hedging_performance_raw.csv still exists for the original delta hedge analysis
DELTA_HEDGING_RAW = "delta_hedging_performance_raw.csv" 
DELTA_VEGA_NO_COST_RAW = "delta_vega_no_cost_raw.csv"
DELTA_VEGA_WITH_COST_RAW = "delta_vega_with_cost_raw.csv"
DELTA_VEGA_SUMMARY = "delta_vega_performance_summary.csv"


def analyze_delta_vega_performance():
    raw_files = [DELTA_VEGA_NO_COST_RAW, DELTA_VEGA_WITH_COST_RAW]
    all_dfs = []
    
    print(f"--- Analyzing Delta-Vega Performance ---")

    # 1. Load and concatenate both raw data files
    for path in raw_files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
                print(f"Loaded {len(df)} rows from {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Warning: Raw file not found at {path}. Skipping.")

    if not all_dfs:
        print("Error: No Delta-Vega raw data files were loaded. Cannot generate summary.")
        return

    # Concatenate all loaded dataframes
    df_raw = pd.concat(all_dfs, ignore_index=True)

    # 2. Group and Aggregate hedging error statistics
    # Grouping by strike type, interval, and fee for the summary
    group_cols_summary = ["strike_type", "interval_days", "brokerage_fee"]

    summary = (
        df_raw.groupby(group_cols_summary, dropna=False)
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            min_mse=("mse", "min"),
            max_mse=("mse", "max"),
            total_observations=("mse", "count")
        )
        .reset_index()
    )

    # 3. Save summary to CSV
    summary.to_csv(DELTA_VEGA_SUMMARY, index=False)

    # 4. Print a human-readable version to the console
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
    print(f"Summary saved to: {DELTA_VEGA_SUMMARY}")
    print(summary)
    print("-" * 40)


def analyze_delta_hedging_performance():
    """Analyzes the original delta hedging performance."""
    raw_path = DELTA_HEDGING_RAW
    summary_path = "delta_hedging_performance_summary.csv"
    
    if not os.path.exists(raw_path):
        print(f"Warning: {raw_path} not found. Skipping Delta Hedging analysis.")
        return

    print(f"--- Analyzing Delta Hedging Performance ---")
    df = pd.read_csv(raw_path)

    group_cols = ["strike_type", "interval_days", "brokerage_fee"]

    summary = (
        df.groupby(group_cols)
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            min_mse=("mse", "min"),
            max_mse=("mse", "max"),
            total_observations=("mse", "count")
        )
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
    print(f"Summary saved to: {summary_path}")
    print(summary)
    print("-" * 40)


def main():
    # If no argument is passed, run both original and delta-vega analyses
    if len(sys.argv) == 1:
        analyze_delta_hedging_performance()
        analyze_delta_vega_performance()
        return
        
    # If an argument is passed, run a specific analysis
    if 'vega' in sys.argv[1].lower():
        analyze_delta_vega_performance()
    else:
        analyze_delta_hedging_performance()

if __name__ == "__main__":
    main()