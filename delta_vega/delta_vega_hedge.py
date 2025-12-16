from enum import Enum
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_greeks,get_implied_volatility

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
    # List of paths to try
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
                print(f"Found file at {path} but failed to read: {e}")
                return None
                
    return None

# Main Hedging Logic
def get_mean_squared_error_delta_vega(interval_days: int, strike_type: StrikeType, maturity_target: str, maturity_rep: str):
     # 1. Load Data
    filename_target = get_filename(maturity_target)
    filename_rep = get_filename(maturity_rep)
    
    df_target = load_csv_robust(filename_target)
    df_rep = load_csv_robust(filename_rep)
    
    if df_target is None or df_rep is None:
        # Only print error if it's strictly missing (avoid cluttering log if just one pair is bad)
        missing = []
        if df_target is None: missing.append(filename_target)
        if df_rep is None: missing.append(filename_rep)
        print(f"Error: Could not find file(s): {', '.join(missing)}")
        return float('nan')

    # 2. Merge Data
    # Convert dates to ensure proper merging
    df_target['Date'] = pd.to_datetime(df_target['Date'])
    df_rep['Date'] = pd.to_datetime(df_rep['Date'])
    
    # Inner join finds only the dates where BOTH options exist (the Overlap)
    df = pd.merge(df_target, df_rep, on=['Date', 'Underlying'], suffixes=('_target', '_rep'))
    
    if df.empty:
        # This happens if the dates don't overlap (e.g. Sept option ends before Nov option starts)
        return float('nan')

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
    
    # Check if selected strike exists in the data columns
    col_target = f"{option_type}{K}_target"
    col_rep = f"{option_type}{K}_rep"
    
    if col_target not in df.columns or col_rep not in df.columns:
        print(f"Strike {K} not found in data columns.")
        return float('nan')
    
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
    # RE Value = alpha * S + eta * C_rep
    hedge_portfolio_value = (alpha * S) + (eta * C_rep_market)
    A_2_sum = 0
    count = 0
    
    # 4. Simulation Loop: Start from the second row (first change)
    # The loop index now represents the day *since* the start (t0)
    for index, row in df.iloc[1:].iterrows():
        #Daily P&L Calculation Always happens
        S_curr = row["Underlying"]
        C_target_curr = row[col_target]
        C_rep_curr = row[col_rep]

        # OP_diff: Change in the Long Option Position
        OP_diff = C_target_curr - C_target_market
        
        # RE_diff: Change in the Short Replicating Portfolio (based on current alpha/eta)
        current_hedge_val = (alpha * S_curr) + (eta * C_rep_curr)
        RE_diff = current_hedge_val - hedge_portfolio_value # <-- This is the key change!
        
        # Total Error (Squared) for THIS day/step (No cost here)
        A_2 = (OP_diff + RE_diff)**2
        
        if np.isfinite(A_2):
            A_2_sum += A_2
            count += 1
        
        #Re-hedging Check Only happens on interval days
        # The re-hedging check should use the *daily index* since the start.
        if (index - 1) % interval_days == 0:
            
            # Update Market State for re-hedging calculation
            C_target_market = C_target_curr
            C_rep_market = C_rep_curr
            S = S_curr
            t_curr_date = row["Date"]
            
            # Time to maturity update
            target_date = datetime.strptime(maturity_target, "%Y-%m-%d") # Re-read to be safe
            rep_date = datetime.strptime(maturity_rep, "%Y-%m-%d")
            T_target = (target_date - t_curr_date).days / 365.0
            T_rep = (rep_date - t_curr_date).days / 365.0
            
            if T_target <= 0 or T_rep <= 0: break
            
            # Re-calculate Volatility & Greeks
            iv_target = get_implied_volatility(C_market=C_target_market, S=S, K=K, T=T_target, r=r)
            iv_rep = get_implied_volatility(C_market=C_rep_market, S=S, K=K, T=T_rep, r=r)
            
            # Handle NaN in implied volatility (Crucial Guardrail)
            if np.isnan(iv_target) or np.isnan(iv_rep):
                # If IV fails, keep old weights (alpha, eta) for next loop and do not update
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
        
        # For ALL days (even non-rehedging days):
        # Update the market values for the P&L calculation in the *next* iteration
        C_target_market = C_target_curr
        C_rep_market = C_rep_curr
        S = S_curr

    if count == 0:
        return float('nan')
        
    mse = A_2_sum / count
    return mse

if __name__ == "__main__":
    valid_pairs = [
        ('2025-09-26', '2025-10-10'), 
        ('2025-09-26', '2025-10-17'), 
        ('2025-10-31', '2025-11-21') 
    ]
    
    interval_days = [1, 2, 7, 10]
    
    print(f"Running Delta-Vega Hedging Simulation...")
    print(f"Checking {len(valid_pairs)} maturity pairs over {len(interval_days)} intervals...")
    
    for interval_day in interval_days:
        print(f"\n--- Interval: {interval_day} Day(s) ---")
        for strike_type in StrikeType:
            for target_mat, rep_mat in valid_pairs:
                mse = get_mean_squared_error_delta_vega(interval_day, strike_type, target_mat, rep_mat)
                
                # Only print if we got a valid result
                if not np.isnan(mse):
                    print(f"Type: {strike_type.value} | Pair: {target_mat} -> {rep_mat} | MSE: {mse:.4f}")