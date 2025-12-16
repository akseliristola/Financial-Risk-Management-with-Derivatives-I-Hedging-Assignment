from enum import Enum
import pandas as pd
import numpy as np
import os
from datetime import datetime

from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_implied_volatility, call_greeks


# Configuration
strike_prices = [165, 170, 185, 190]

class StrikeType(Enum):
    ATM = "ATM"
    OTM = "OTM"
    ITM = "ITM"

# Helper Functions
def calculate_portfolio_weights(delta_bs, vega_bs, delta_rep, vega_rep):
    """
    Calculates the weights for the replicating portfolio.
    eta: Quantity of Replicating Options (to hedge Vega)
    alpha: Quantity of Underlying Asset (to hedge remaining Delta)
    """
    if vega_rep == 0:
        eta = 0.0
    else:
        eta = -vega_bs / vega_rep
    
    alpha = -delta_bs - (eta * delta_rep)
    return alpha, eta

def get_filename(maturity_str):
    """
    Constructs the filename based on the maturity date.
    Format: options_data_YYYY_M_D.csv (single digit for month/day)
    """
    dt = datetime.strptime(maturity_str, "%Y-%m-%d")
    filename_date = f"{dt.year}_{dt.month}_{dt.day}"
    return f"options_data_{filename_date}.csv"

def load_csv_robust(filename):
    """
    Tries to load the CSV from multiple common locations.
    Returns the DataFrame or None if not found.
    """
    paths_to_try = [
        f"data/{filename}",  
        filename,            
        f"../data/{filename}"
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                # print(f"Found file at {path} but failed to read: {e}")
                return None
                
    return None

# Main Hedging Logic - Returns MSE and Iterations Count (n)
def get_mean_squared_error_delta_vega(interval_days: int, strike_type: StrikeType, maturity_target: str, maturity_rep: str):
     # 1. Load Data
    filename_target = get_filename(maturity_target)
    filename_rep = get_filename(maturity_rep)
    
    df_target = load_csv_robust(filename_target)
    df_rep = load_csv_robust(filename_rep)
    
    if df_target is None or df_rep is None:
        return float('nan'), 0

    # 2. Merge Data
    df_target['Date'] = pd.to_datetime(df_target['Date'])
    df_rep['Date'] = pd.to_datetime(df_rep['Date'])
    df = pd.merge(df_target, df_rep, on=['Date', 'Underlying'], suffixes=('_target', '_rep'))
    
    if df.empty:
        return float('nan'), 0

    first_row = df.iloc[0]
    S0 = first_row["Underlying"]
    
    # 3. Select Strike
    if strike_type == StrikeType.ATM:
        K = min(strike_prices, key=lambda k: abs(k - S0))
    elif strike_type == StrikeType.ITM:
        K = max(strike_prices) 
    elif strike_type == StrikeType.OTM:
        K = min(strike_prices)
    
    option_type = "C"
    r = 0.06
    
    col_target = f"{option_type}{K}_target"
    col_rep = f"{option_type}{K}_rep"
    
    if col_target not in df.columns or col_rep not in df.columns:
        return float('nan'), 0
    
    # Initial Setup
    t_current = first_row["Date"]
    target_date = datetime.strptime(maturity_target, "%Y-%m-%d")
    rep_date = datetime.strptime(maturity_rep, "%Y-%m-%d")
    
    T_target = (target_date - t_current).days / 365.0
    T_rep = (rep_date - t_current).days / 365.0
    
    S = S0
    C_target_market = first_row[col_target]
    C_rep_market = first_row[col_rep]
    
    # Initial Greeks
    iv_target = get_implied_volatility(C_market=C_target_market, S=S, K=K, T=T_target, r=r)
    iv_rep = get_implied_volatility(C_market=C_rep_market, S=S, K=K, T=T_rep, r=r)
    
    greeks_target = call_greeks(S, K, T_target, r, iv_target)
    greeks_rep = call_greeks(S, K, T_rep, r, iv_rep)
    
    # Initial Weights
    alpha, eta = calculate_portfolio_weights(
        greeks_target['delta'], greeks_target['vega'],
        greeks_rep['delta'], greeks_rep['vega']
    )
    
    # Initial Portfolio Value
    hedge_portfolio_value = (alpha * S) + (eta * C_rep_market)
    
    A_2_sum = 0
    count = 0
    n = 1 # Count initial position (for iterations count)

    # 4. Simulation Loop: Start from the second row (first change)
    for index, row in df.iloc[1:].iterrows():
        #Daily P&L Calculation Always happens
        S_curr = row["Underlying"]
        C_target_curr = row[col_target]
        C_rep_curr = row[col_rep]

        # OP_diff: Change in the Long Option Position
        OP_diff = C_target_curr - C_target_market
        
        # RE_diff: Change in the Short Replicating Portfolio
        current_hedge_val = (alpha * S_curr) + (eta * C_rep_curr)
        RE_diff = current_hedge_val - hedge_portfolio_value
        
        # Total Error (Squared) for THIS day/step (No cost here)
        A_2 = (OP_diff + RE_diff)**2
        
        if np.isfinite(A_2):
            A_2_sum += A_2
            count += 1
        
        #Re-hedging Check Only happens on interval days
        if (index - 1) % interval_days == 0:
            
            # Update Market State for re-hedging calculation
            C_target_market = C_target_curr
            C_rep_market = C_rep_curr
            S = S_curr
            t_curr_date = row["Date"]
            
            # Time to maturity update
            target_date = datetime.strptime(maturity_target, "%Y-%m-%d")
            rep_date = datetime.strptime(maturity_rep, "%Y-%m-%d")
            T_target = (target_date - t_curr_date).days / 365.0
            T_rep = (rep_date - t_curr_date).days / 365.0
            
            if T_target <= 0 or T_rep <= 0: break
            
            # Re-calculate Volatility & Greeks
            iv_target = get_implied_volatility(C_market=C_target_market, S=S, K=K, T=T_target, r=r)
            iv_rep = get_implied_volatility(C_market=C_rep_market, S=S, K=K, T=T_rep, r=r)
            
            # Handle NaN in implied volatility (Crucial Guardrail)
            if np.isnan(iv_target) or np.isnan(iv_rep):
                continue 

            g_target = call_greeks(S, K, T_target, r, iv_target)
            g_rep = call_greeks(S, K, T_rep, r, iv_rep)
            
            # Re-balance Weights
            alpha, eta = calculate_portfolio_weights(
                g_target['delta'], g_target['vega'],
                g_rep['delta'], g_rep['vega']
            )
            
            # Reset Hedge Portfolio Value with new weights at current price
            hedge_portfolio_value = (alpha * S) + (eta * C_rep_market)
            n += 1 # Count re-hedging event
        
        # For ALL days (even non-rehedging days):
        C_target_market = C_target_curr
        C_rep_market = C_rep_curr
        S = S_curr

    if count == 0:
        return float('nan'), n
        
    mse = A_2_sum / count
    return mse, n

if __name__ == "__main__":
    valid_pairs = [
        ('2025-09-26', '2025-10-10'), 
        ('2025-09-26', '2025-10-17'), 
        ('2025-10-31', '2025-11-21') 
    ]
    
    interval_days_list = [1, 2, 7, 10]
    brokerage_fee = 0.0 # Set to 0.0 for the 'no cost' file
    
    output_path = "delta_vega_no_cost_raw.csv"
    records = []
    
    print(f"Running Delta-Vega Hedging Simulation (No Cost)...")
    
    for interval_day in interval_days_list:
        for strike_type in StrikeType:
            for target_mat, rep_mat in valid_pairs:
                mse, iterations = get_mean_squared_error_delta_vega(
                    interval_day, 
                    strike_type, 
                    target_mat, 
                    rep_mat
                )
                
                if not np.isnan(mse):
                    records.append(
                        {
                            "maturity": target_mat, 
                            "interval_days": interval_day,
                            "strike_type": strike_type.value,
                            "brokerage_fee": brokerage_fee,
                            "mse": mse,
                            "iterations": iterations
                        }
                    )
    
    # Turn into DataFrame and write CSV
    df_results = pd.DataFrame(records)
    df_results.to_csv(output_path, index=False)
    
    print(f"Saved {len(df_results)} rows to '{output_path}'.")