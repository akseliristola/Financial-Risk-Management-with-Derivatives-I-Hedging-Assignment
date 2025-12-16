from enum import Enum
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_implied_volatility, get_call_option_delta

strike_prices=[165,170,185,190]
brokerage_fees = [0, 0.05]

def calculate_short_position_value(delta,S):
    return delta*S


class StrikeType(Enum):
    ATM = "ATM"
    OTM = "OTM"
    ITM = "ITM"

def get_mean_squared_error(interval_days:int=7,strike_type:StrikeType=StrikeType.ATM,maturity:str="2025-11-14",brokerage_fee:float=0.0):
    df=pd.read_csv(f"../data/options_data_{maturity.replace('-', '_')}.csv")

    first_row=df.iloc[0]
    
    S0 = first_row["Underlying"]
    if strike_type == StrikeType.ATM:
        K = min(strike_prices, key=lambda k: abs(k - S0))
    elif strike_type == StrikeType.ITM:
        K = max(strike_prices)
    elif strike_type == StrikeType.OTM:
        K = min(strike_prices)
    option_type = "C"
    
    r=0.06
    S=first_row["Underlying"]
    T=(datetime.strptime(maturity, "%Y-%m-%d")-datetime.strptime(first_row["Date"], "%Y-%m-%d")).days/365
    implied_volatility=get_implied_volatility(C_market=first_row[f"{option_type}{K}"], S=S, K=K, T=T, r=r)
    delta = get_call_option_delta(S=S, K=K, T=T, r=r, sigma=implied_volatility)

    C_market=first_row[f"{option_type}{K}"]
    S=first_row["Underlying"]
    
    short_position_value=calculate_short_position_value(delta,S)
    short_position=short_position_value/S
    A_2_sum=0
    n = 1  # Count initial position (for iterations count)
    
    C_market_prev = C_market
    S_prev = S
    short_position_prev = short_position
    
    for index, row in df.iloc[1:].iterrows():
        C_market_now = row[f"{option_type}{K}"]
        S_now = row["Underlying"]
        
        # Calculate error on every day
        OP_diff = C_market_now - C_market_prev
        RE_diff = -short_position_prev * (S_now - S_prev)  # Negative because we're SHORT the stock
        A_2 = (OP_diff + RE_diff)**2
        A_2_sum += A_2  # add error despite no re-hedging
        
        if (index-1)%interval_days==0:  # re-hedge here
            T_t = (datetime.strptime(maturity, "%Y-%m-%d")-datetime.strptime(row["Date"], "%Y-%m-%d")).days/365
            
            implied_volatility=get_implied_volatility(C_market=C_market_now, S=S_now, K=K, T=T_t, r=r)
            
            delta_new = get_call_option_delta(S=S_now, K=K, T=T_t, r=r, sigma=implied_volatility)
            short_position_new = delta_new  # number of shares to short (positive)
            short_position_value_new = short_position_new * S_now
            
            # Calculate transaction cost for re-hedging (similar to delta_gamma)
            # transaction_size is the dollar value of the change in position
            transaction_size = abs(short_position_new - short_position_prev) * S_now
            transaction_cost = transaction_size * brokerage_fee
            A_2_sum += transaction_cost**2  # added as "hedging error" because costs incurred are similar
            
            short_position_prev = short_position_new
            n += 1  # Count re-hedging event
        
        C_market_prev = C_market_now
        S_prev = S_now
    mse=A_2_sum/(n-1) if n > 1 else 0.0
    return mse, n


if __name__ == "__main__":
    maturities = [
        '2025-9-12',
        '2025-9-19',
        '2025-9-26',
        '2025-10-10',
        '2025-10-17',
        '2025-10-24',
        '2025-10-31',
        '2025-11-14',
        '2025-11-21',
        '2025-11-28',
    ]
    
    interval_days_list = [1, 2, 7, 10]
    
    output_path = "delta_hedging_performance_raw.csv"
    records = []
    
    for interval_day in interval_days_list:
        for strike_type in StrikeType:
            for maturity in maturities:
                for fee in brokerage_fees:
                    mse, iterations = get_mean_squared_error(
                        interval_day,
                        strike_type,
                        maturity,
                        fee
                    )
                    
                    records.append(
                        {
                            "maturity": maturity,
                            "interval_days": interval_day,
                            "strike_type": strike_type.value,
                            "brokerage_fee": fee,
                            "mse": mse,
                            "iterations": iterations
                        }
                    )
    
    # Turn into DataFrame and write CSV
    df_results = pd.DataFrame(records)
    df_results.to_csv(output_path, index=False)
    
    print(f"Saved {len(df_results)} rows to '{output_path}'.")
