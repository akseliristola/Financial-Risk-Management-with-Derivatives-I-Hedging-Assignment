from enum import Enum
import pandas as pd
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_implied_volatility, get_call_option_delta

from utils import (
    black_scholes_call_price,
    get_implied_volatility,
    get_call_option_delta,
    get_option_gamma,
    GAMMA_EPS,
    gamma_hedging_strike,
)

brokerage_fees = [0, 0.02] # use for iterating witha and without fees 

strikes=[165,170,185,190]

def delta_gamma_update(C1, C2, S, K1, K2, T, r):
    
    GAMMA_EPS = 1e-10
    fallback = 0
    # Implied vols
    sigma1 = get_implied_volatility(C_market=C1, S=S, K=K1, T=T, r=r)
    sigma2 = get_implied_volatility(C_market=C2, S=S, K=K2, T=T, r=r)

    # Greeks for C1 and C2
    delta1 = get_call_option_delta(S=S, K=K1, T=T, r=r, sigma=sigma1)
    gamma1 = get_option_gamma(S=S, K=K1, T=T, r=r, sigma=sigma1)

    delta2 = get_call_option_delta(S=S, K=K2, T=T, r=r, sigma=sigma2)
    gamma2 = get_option_gamma(S=S, K=K2, T=T, r=r, sigma=sigma2)

    # Solve for y (units of C2) and x (units of stock) to hedge 1 unit of C1
    if abs(gamma1) < GAMMA_EPS:
        y = 0.0
        x = -delta1
    elif abs(gamma2) < GAMMA_EPS:
        # C2 has basically no gamma -> can't gamma-hedge => pure delta hedge
        y = 0.0
        x = -delta1
        fallback = 1
        print("Option C2 has 0 gamma, defaulting to simple delta-hedge")
    else:
        y = - gamma1 / gamma2 #buy c1 and short others
        x = - (delta1 + y * delta2)

    return x, y, fallback
    #Now our OP = x*S+C1+y*C2


class StrikeType(Enum):
    ATM = "ATM"
    OTM = "OTM"
    ITM = "ITM"

def get_mean_squared_error(interval_days, strike_type ,maturity, brokerage_fee):
    df=pd.read_csv(f"../data/options_data_{maturity.replace('-', '_')}.csv")

    first_row=df.iloc[0]
    A_squared_sum = 0

    gamma_failures = 0 #Track ratio of defaulting to delta-hedge
    n = 0

    S_prev = first_row["Underlying"]
    r=0.06
    T_prev = (datetime.strptime(maturity, "%Y-%m-%d") - datetime.strptime(first_row["Date"], "%Y-%m-%d")).days / 365

    if strike_type == StrikeType.ATM:
        K1 = min(strikes, key=lambda k: abs(k - S_prev ))
    elif strike_type == StrikeType.ITM:
        K1 = max(strikes)
    elif strike_type == StrikeType.OTM:
        K1 = min(strikes)
    option_type = "C" 

    C1_col = f"{option_type}{K1}"

    # Choose C2: another call, same maturity, different strike
    K2_curr = gamma_hedging_strike(
        row=first_row,
        K1=K1,
        S=S_prev,
        T=T_prev,
        r=r,
        strikes=strikes,
        option_type=option_type,
    )

    if K2_curr is None:
        available_strikes_for_c2 = [k for k in strikes if k != K1]
        K2_curr = min(available_strikes_for_c2, key=lambda k: abs(k - S_prev))

    C2_col_curr = f"{option_type}{K2_curr}"

    C1_price_prev = first_row[C1_col]
    C2_price_prev = first_row[C2_col_curr]

    x, y, fallback = delta_gamma_update(
        C1=C1_price_prev,
        C2=C2_price_prev,
        S=S_prev,
        K1=K1,
        K2=K2_curr,
        T=T_prev,
        r=r,
    )
    gamma_failures += fallback
    n +=1
    

    for index, row in df.iloc[1:].iterrows():
        
        C1_price_now = row[f"{option_type}{K1}"]
        C2_price_now = row[f"{option_type}{K2_curr}"]
        S_price_now = row["Underlying"]

        dC1 = C1_price_now - C1_price_prev #long call option, if C1_t > C1_t-1 we are losing
        dS = x * (S_price_now - S_prev)
        dC2 = y * (C2_price_now - C2_price_prev)
        dRE = dS + dC2
        A_squared = (dC1 + dRE)**2 #dRE is already opposite to dOP. If dRE is negative when stock goes up and dOP is posi 
        
        A_squared_sum += A_squared #add error despite no re-hedging
        
        if (index-1)%interval_days==0: #re-hedge here

            T_t =  (datetime.strptime(maturity, "%Y-%m-%d") - datetime.strptime(row["Date"], "%Y-%m-%d")).days / 365
            
            # Choose hedging option C2 
            # Calculate C1 gamma & delta.
            # Calculate new amount x for S and calculate d_x to know how much to sell / buy
            n += 1

            K2_next = gamma_hedging_strike(
                row=row,
                K1=K1,
                S=S_price_now,
                T=T_t,
                r=r,
                strikes=strikes,
                option_type=option_type,
            )
            if K2_next is None:
                available_strikes_for_c2 = [k for k in strikes if k != K1]
                K2_next = min(available_strikes_for_c2, key=lambda k: abs(k - S_price_now))
            
            close_short = y if K2_next != K2_curr else 0 #close short on y units of old hedging option if changed (increases our buying of C2 by y if done)
            closing_cost = close_short * C2_price_now

            C2_t_next = row[f"{option_type}{K2_next}"] #price of the new option for next hedging interval

            x_new, y_new, fallback = delta_gamma_update(
                C1=C1_price_now,
                C2=C2_t_next,
                S=S_price_now,
                K1=K1,
                K2=K2_next,
                T=T_t,
                r=r,
            )
            gamma_failures += fallback

            transaction_size = abs(x_new - x) * S_price_now + abs(closing_cost) + abs(y_new - y + close_short) * C2_t_next
            transaction_cost = transaction_size * brokerage_fee

            A_squared_sum += transaction_cost**2 #added as "hedging error" because costs incurred are similar

            K2_curr = K2_next
            C2_price_now = C2_t_next
            
        C1_price_prev = C1_price_now
        C2_price_prev = C2_price_now
        S_prev = S_price_now

        if np.isnan(A_squared_sum):
            print(f"Maturity: {maturity}, date: {row['Date']}, C1: {C1_col} priced at {C1_price_prev}, C2: {K2_curr} priced at {C2_price_prev}")
            raise ValueError("A_squared_sum became NaN - check data for missing prices.")


    return A_squared_sum/(n-1), gamma_failures, n
'''
if __name__ == "__main__":
    fail_tot = 0
    iter_tot = 0
    maturities = ['2025-9-12','2025-9-19','2025-9-26','2025-11-28','2025-11-21','2025-10-31','2025-10-24', '2025-11-14', '2025-10-10','2025-10-17']
    interval_days=[1,2,7,10]
    for interval_day in interval_days:
        for strike_type in StrikeType:
            for maturity in maturities:
                for fee in brokerage_fees:
                    mse, gamma_failures, n = get_mean_squared_error(interval_day,strike_type,maturity, fee)
                    print(f"MSE with {interval_day} day interval, {strike_type} strike type, {maturity} maturity and {fee*100}% fee:",mse)
                    print(f"Ratio of defaults to pure delta-hedge: {gamma_failures} out of {iterations} runs")
                    fail_tot += gamma_failures
                    iter_tot += n
                print("--------------------------------")
    print(f"Ratio of defaults to pure delta-hedge: {fail_tot} out of {iter_tot} runs")
'''
if __name__ == "__main__":
    # Define the expiries you want to treat as 10 repetitions
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

    # Hedging intervals (in days) you want scenarios for
    interval_days_list = [1, 2, 7, 10]

    output_path = "gamma_hedging_performance_raw.csv"
    records = []

    for interval_day in interval_days_list:
        for strike_type in StrikeType:  # ATM, ITM, OTM
            for maturity in maturities:
                for fee in brokerage_fees:  # [0, 0.02]
                    mse, gamma_failures, iterations = get_mean_squared_error(
                        interval_day,
                        strike_type,
                        maturity,
                        fee
                    )
                    default_ratio = (
                        gamma_failures / iterations if iterations > 0 else 0.0
                    )

                    records.append(
                        {
                            "maturity": maturity,
                            "interval_days": interval_day,
                            "strike_type": strike_type.value,  # store as string
                            "brokerage_fee": fee,
                            "mse": mse,
                            "gamma_failures": gamma_failures,
                            "iterations": iterations,
                            "default_ratio": default_ratio,
                        }
                    )

    # Turn into DataFrame and write CSV
    df_results = pd.DataFrame(records)
    df_results.to_csv(output_path, index=False)

    print(f"Saved {len(df_results)} rows to '{output_path}'.")