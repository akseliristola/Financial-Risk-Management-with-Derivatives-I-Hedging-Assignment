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

strike_prices = [165, 170, 185, 190]
TRANSACTION_COST_RATE = 0.05  # 5% cost on traded value

class StrikeType(Enum):
    ATM = "ATM"
    OTM = "OTM"
    ITM = "ITM"

# Helper Functions
def calculate_portfolio_weights(delta_bs, vega_bs, delta_rep, vega_rep):
    if vega_rep == 0:
        eta = 0.0
    else:
        eta = -vega_bs / vega_rep
    
    alpha = -delta_bs - (eta * delta_rep)
    return alpha, eta

def get_filename(maturity_str):
    dt = datetime.strptime(maturity_str, "%Y-%m-%d")
    filename_date = f"{dt.year}_{dt.month}_{dt.day}"
    return f"options_data_{filename_date}.csv"

def load_csv_robust(filename):
    paths_to_try = [f"data/{filename}", filename, f"../data/{filename}"]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                continue
    return None

# Main Hedging Logic - Returns MSE and Iterations Count (n)
def run_simulation_with_costs(interval_days: int, strike_type: StrikeType, maturity_target: str, maturity_rep: str):
    
    #Load & Merge Data
    filename_target = get_filename(maturity_target)
    filename_rep = get_filename(maturity_rep)
    
    df_target = load_csv_robust(filename_target)
    df_rep = load_csv_robust(filename_rep)
    
    if df_target is None or df_rep is None:
        return None, 0

    df_target['Date'] = pd.to_datetime(df_target['Date'])
    df_rep['Date'] = pd.to_datetime(df_rep['Date'])
    
    df = pd.merge(df_target, df_rep, on=['Date', 'Underlying'], suffixes=('_target', '_rep'))
    
    if df.empty:
        return None, 0

    first_row = df.iloc[0]
    S0 = first_row["Underlying"]
    
    #Select Strike
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
    
    if col_target not in df.columns:
        return None, 0
    
    # Initial Setup
    t_current = first_row["Date"]
    target_date = datetime.strptime(maturity_target, "%Y-%m-%d")
    rep_date = datetime.strptime(maturity_rep, "%Y-%m-%d")
    
    T_target = (target_date - t_current).days / 365.0
    T_rep = (rep_date - t_current).days / 365.0
    
    S = S0
    C_target_market = first_row[col_target]
    C_rep_market = first_row[col_rep]
    
    # Calculate Initial Weights
    iv_target = get_implied_volatility(C_target_market, S, K, T_target, r)
    iv_rep = get_implied_volatility(C_rep_market, S, K, T_rep, r)
    
    g_target = call_greeks(S, K, T_target, r, iv_target)
    g_rep = call_greeks(S, K, T_rep, r, iv_rep)
    
    # Current Positions
    alpha_curr, eta_curr = calculate_portfolio_weights(
        g_target['delta'], g_target['vega'],
        g_rep['delta'], g_rep['vega']
    )
    
    # Track accumulated values using the just-calculated initial positions
    hedge_portfolio_value = (alpha_curr * S) + (eta_curr * C_rep_market)
    
    A_2_sum = 0
    total_transaction_costs = 0.0
    count = 0
    n = 1 # Count initial position (for iterations count)
    
    for index, row in df.iloc[1:].iterrows():
        
        #Daily P&L Calculation (Always happens)
        S_curr = row["Underlying"]
        C_target_curr = row[col_target]
        C_rep_curr = row[col_rep]
        
        OP_diff = C_target_curr - C_target_market
        
        # Change in value of our EXISTING positions
        current_hedge_val = (alpha_curr * S_curr) + (eta_curr * C_rep_curr)
        RE_diff = current_hedge_val - hedge_portfolio_value
        
        step_cost = 0.0 # Initialize cost for the day
        
        #Re-hedging Check Only happens on interval days
        if (index - 1) % interval_days == 0:
            
            # Calculate New Weights (Rebalancing) ---
            t_curr_date = row["Date"]
            target_date = datetime.strptime(maturity_target, "%Y-%m-%d")
            rep_date = datetime.strptime(maturity_rep, "%Y-%m-%d")
            T_target = (target_date - t_curr_date).days / 365.0
            T_rep = (rep_date - t_curr_date).days / 365.0
            
            if T_target <= 0 or T_rep <= 0: break

            iv_target = get_implied_volatility(C_target_curr, S_curr, K, T_target, r)
            iv_rep = get_implied_volatility(C_rep_curr, S_curr, K, T_rep, r)

            # Handle NaN in implied volatility
            if np.isnan(iv_target) or np.isnan(iv_rep):
                 alpha_new, eta_new = alpha_curr, eta_curr # Keep old weights
            else:
                g_target = call_greeks(S_curr, K, T_target, r, iv_target)
                g_rep = call_greeks(S_curr, K, T_rep, r, iv_rep)
                
                alpha_new, eta_new = calculate_portfolio_weights(
                    g_target['delta'], g_target['vega'],
                    g_rep['delta'], g_rep['vega']
                )

                # Calculate Transaction Costs (using old vs. new position)
                stock_traded_val = abs(alpha_new - alpha_curr) * S_curr
                option_traded_val = abs(eta_new - eta_curr) * C_rep_curr
                
                step_cost = (stock_traded_val + option_traded_val) * TRANSACTION_COST_RATE
                total_transaction_costs += step_cost
            
            # Update State for Next Loop
            alpha_curr = alpha_new
            eta_curr = eta_new
            n += 1 # Count re-hedging event
            
            # Reset Hedge Portfolio Value for the NEW positions
            hedge_portfolio_value = (alpha_curr * S_curr) + (eta_curr * C_rep_curr)

        #Compute Total Error A_i
        # A_i = (Gain in Option) + (Gain in Hedge) - (Transaction Cost)
        A_i = (OP_diff + RE_diff) - step_cost
        
        if np.isfinite(A_i):
            A_2_sum += A_i**2
            count += 1
        
        # For ALL days: Update the market values for the P&L calculation in the *next* iteration
        C_target_market = C_target_curr
        C_rep_market = C_rep_curr

    if count == 0:
        return None, n
        
    mse = A_2_sum / count
    return mse, n

if __name__ == "__main__":
    valid_pairs = [
        ('2025-09-26', '2025-10-10'), 
        ('2025-09-26', '2025-10-17'), 
        ('2025-10-31', '2025-11-21') 
    ]
    
    interval_days_list = [1, 2, 7, 10]
    brokerage_fee = TRANSACTION_COST_RATE # Set to 0.05 for this file
    
    output_path = "delta_vega_with_cost_raw.csv"
    records = []
    
    print(f"Running Delta-Vega Hedging Simulation (With {brokerage_fee*100}% Cost)...")
    
    for interval_day in interval_days_list:
        for strike_type in StrikeType:
            for target_mat, rep_mat in valid_pairs:
                mse, iterations = run_simulation_with_costs(
                    interval_day, 
                    strike_type, 
                    target_mat, 
                    rep_mat
                )
                
                if mse is not None:
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