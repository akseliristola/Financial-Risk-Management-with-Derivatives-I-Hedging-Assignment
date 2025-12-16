import numpy as np
from scipy.stats import norm

GAMMA_EPS = 1e-6

def black_scholes_call_price(S, K, T, r, sigma):
    # Handle edge cases
    if T <= 0:
        # If time to expiration is zero or negative, return intrinsic value
        return max(S - K, 0)
    if sigma <= 0:
        # If volatility is zero or negative, return intrinsic value
        return max(S - K * np.exp(-r * T), 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    

def get_implied_volatility(C_market, S, K, T, r, tol=1e-6, max_iter=100):
    # Handle edge cases
    if T <= 0:
        # If time to expiration is zero or negative, return a default volatility
        return 0.2
    
    # Calculate intrinsic value
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if C_market <= intrinsic:
        # Market price is at or below intrinsic value, return small volatility
        return 1e-6
    
    sigma = 0.2  # initial guess
    vega_min = 1e-10  # minimum vega threshold to avoid division by zero
    sigma_min = 1e-6  # minimum volatility to avoid division by zero
    
    for i in range(max_iter):
        # Ensure sigma stays positive
        sigma = max(sigma, sigma_min)
        
        price = black_scholes_call_price(S, K, T, r, sigma)

        # Calculate vega with safety checks
        if sigma > sigma_min and T > 0:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
        else:
            vega = vega_min
        
        diff = price - C_market
        if abs(diff) < tol:
            return sigma
        
        # Handle case where vega is too small (near zero)
        if abs(vega) < vega_min:
            # Use a small fixed step size when vega is too small
            sigma -= np.sign(diff) * 0.01
        else:
            sigma -= diff / vega
        
        # Ensure sigma stays in reasonable bounds
        sigma = max(sigma_min, min(sigma, 5.0))
    
    return sigma

def get_call_option_delta(S, K, T, r, sigma, t=0):
    tau = T - t  # time to maturity
    
    # Handle edge cases
    if tau <= 0:
        # If time to expiration is zero or negative, return intrinsic delta
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        # If volatility is zero, return intrinsic delta
        return 1.0 if S > K * np.exp(-r * tau) else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    delta = norm.cdf(d1)
    return delta



def get_option_gamma(S, K, T, r, sigma, t=0):
    tau = T - t  # time to maturity
    # Handle edge cases
    if tau <= 0 or sigma <= 0:
        # At or past expiry, or zero vol -> gamma effectively 0
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(tau))

    return gamma




def gamma_hedging_strike(row, K1, S, T, r, strikes, option_type="C"):

    # Sort candidate strikes by distance to the underlying price,
    # ignoring the original strike K1.
    candidate_strikes = sorted(
        (k for k in strikes if k != K1),
        key=lambda k: abs(k - S)
    )

    best_strike = None
    best_gamma_abs = -1.0  # track "least bad" gamma in case all are tiny

    for k in candidate_strikes:
        price_key = f"{option_type}{k}"

        if price_key not in row:
            continue

        C_market = row[price_key]
        if C_market is None or np.isnan(C_market):
            continue

        sigma = get_implied_volatility(C_market, S, k, T, r)

        gamma = get_option_gamma(S, k, T, r, sigma)

        g_abs = abs(gamma)

        # Prefer the first strike with meaningful gamma
        if g_abs >= GAMMA_EPS:
            return k

        # Otherwise, remember the best we’ve seen so far
        if g_abs > best_gamma_abs:
            best_gamma_abs = g_abs
            best_strike = k

    # If all gammas were extremely small or data was sparse,
    # fall back to the best candidate we found (or None).
    return best_strike


def call_vega(S, K, T, r, sigma):
    tau = T
    if tau <= 0:
        return 0.0

    if sigma <= 0:
        sigma = 1e-4

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)


def call_greeks(S, K, T, r, sigma):
    price = black_scholes_call_price(S, K, T, r, sigma)
    delta = get_call_option_delta(S, K, T, r, sigma)
    vega  = call_vega(S, K, T, r, sigma)
    return {
        'price': price,
        'delta': delta,
        'vega':  vega
    }