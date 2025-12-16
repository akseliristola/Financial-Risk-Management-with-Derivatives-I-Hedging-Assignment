# Financial Risk Management with Derivatives I - Hedging Strategies Project

This project is part of the **Financial Risk Management with Derivatives I** course at Aalto University. It implements and compares different options hedging strategies to manage the risk of option positions.

## Project Overview

The project evaluates three different hedging strategies for managing option positions:

1. **Delta Hedging** - Hedges price risk using the underlying asset
2. **Delta-Gamma Hedging** - Hedges both price and convexity risk using the underlying asset and another option
3. **Delta-Vega Hedging** - Hedges both price and volatility risk using the underlying asset and a replicating option with different maturity

Each strategy is tested across various scenarios to measure hedging effectiveness using Mean Squared Error (MSE) as the performance metric.

## Project Structure

```
./
├── data/                          # Options market data
│   ├── options_data_2025_9_12.csv
│   ├── options_data_2025_9_19.csv
│   ├── options_data_2025_9_26.csv
│   ├── options_data_2025_10_10.csv
│   ├── options_data_2025_10_17.csv
│   ├── options_data_2025_10_24.csv
│   ├── options_data_2025_10_31.csv
│   ├── options_data_2025_11_14.csv
│   ├── options_data_2025_11_21.csv
│   └── options_data_2025_11_28.csv
├── delta/                         # Delta hedging strategy
│   ├── delta_heding.py           # Main delta hedging implementation
│   ├── analyze_hedging_performance.py
│   ├── delta_hedging_performance_raw.csv
│   └── delta_hedging_performance_summary.csv
├── delta_gamma/                   # Delta-Gamma hedging strategy
│   ├── delta_gamma.py            # Main delta-gamma hedging implementation
│   ├── analyze_hedging_performance.py
│   ├── gamma_hedging_performance_raw.csv
│   └── gamma_hedging_performance_summary.csv
├── delta_vega/                    # Delta-Vega hedging strategy
│   ├── delta_vega_hedge.py       # Delta-Vega hedging without transaction costs
│   ├── delta_vega_cost.py        # Delta-Vega hedging with transaction costs
│   ├── analyze_delta_vega_performance.py
│   ├── delta_vega_no_cost_raw.csv
│   ├── delta_vega_with_cost_raw.csv
│   └── delta_vega_performance_summary.csv
├── utils.py                       # Shared utilities (Black-Scholes, Greeks, etc.)
├── option_ric_tools.py            # Tools for fetching option data from Refinitiv
└── data_fetching.ipynb            # Jupyter notebook for data collection
```

## Data

The project uses historical options market data stored in CSV files in the `data/` directory. Each file contains:

- **Date**: Trading date
- **Underlying**: Spot price of the underlying asset
- **Call Options**: Prices for call options at different strikes (C165, C170, C185, C190, C205)
- **Put Options**: Prices for put options at different strikes (P165, P170, P185, P190, P205)

The data covers multiple maturity dates from September 2025 to November 2025, allowing for testing across different time horizons.

**Data Collection**: The `data_fetching.ipynb` notebook and `option_ric_tools.py` module provide utilities for fetching options market data from Refinitiv Data Platform. The `option_ric_tools.py` module includes functions to construct option RICs (Reuters Instrument Codes) for various exchanges including OPRA, Eurex, HKEX, OSE, and IEU.

## Hedging Strategies

### 1. Delta Hedging (`delta/delta_heding.py`)

**Concept**: Hedges the price sensitivity (delta) of an option position by taking an opposite position in the underlying asset.

**How it works**:

- Short 1 unit of a call option
- Long `delta` units of the underlying stock to hedge
- Re-hedge periodically to maintain delta neutrality

**Parameters tested**:

- Strike types: ATM (At-The-Money), ITM (In-The-Money), OTM (Out-The-Money)
- Re-hedging intervals: 1, 2, 7, 10 days
- Brokerage fees: 0%, 2%
- Multiple maturity dates

**Running the strategy**:

```bash
cd delta
python delta_heding.py
```

This generates `delta_hedging_performance_raw.csv` with MSE results for all parameter combinations.

### 2. Delta-Gamma Hedging (`delta_gamma/delta_gamma.py`)

**Concept**: Hedges both delta (price sensitivity) and gamma (convexity) by using the underlying asset and another option.

**How it works**:

- Short 1 unit of call option C1
- Long `x` units of underlying stock
- Long/short `y` units of another call option C2 (selected to hedge gamma)
- The portfolio is constructed to be both delta-neutral and gamma-neutral

**Parameters tested**:

- Same as delta hedging (strike types, intervals, fees, maturities)
- Additionally tracks gamma hedging failures (when C2 has insufficient gamma)

**Running the strategy**:

```bash
cd delta_gamma
python delta_gamma.py
```

This generates `gamma_hedging_performance_raw.csv` with MSE results and gamma failure statistics.

### 3. Delta-Vega Hedging (`delta_vega/`)

**Concept**: Hedges both delta (price sensitivity) and vega (volatility sensitivity) using the underlying asset and a replicating option with different maturity.

**How it works**:

- Long 1 unit of target call option (with maturity T1)
- Short replicating portfolio: `alpha` units of underlying + `eta` units of replicating call option (with maturity T2)
- Portfolio weights are calculated to neutralize both delta and vega

**Two implementations**:

- `delta_vega_hedge.py`: Basic implementation without transaction costs
- `delta_vega_cost.py`: Implementation with 5% transaction costs on re-hedging

**Parameters tested**:

- Strike types: ATM, ITM, OTM
- Re-hedging intervals: 1, 7 days
- Valid maturity pairs: (2025-09-26, 2025-10-10), (2025-09-26, 2025-10-17), (2025-10-31, 2025-11-21)

**Running the strategies**:

```bash
cd delta_vega
python delta_vega_hedge.py    # Without transaction costs
python delta_vega_cost.py      # With transaction costs
```

This generates `delta_vega_no_cost_raw.csv` and `delta_vega_with_cost_raw.csv` with MSE results for all parameter combinations.

## Utilities (`utils.py`)

The `utils.py` module provides shared functionality used by all strategies:

- **Black-Scholes pricing**: `black_scholes_call_price()`
- **Implied volatility calculation**: `get_implied_volatility()` (using Newton-Raphson method)
- **Greeks calculation**:
  - `get_call_option_delta()` - Delta (price sensitivity)
  - `get_option_gamma()` - Gamma (convexity)
  - `call_vega()` - Vega (volatility sensitivity)
  - `call_greeks()` - Returns all Greeks in a dictionary
- **Gamma hedging strike selection**: `gamma_hedging_strike()` - Selects optimal strike for gamma hedging

## Performance Analysis

Each strategy directory contains an `analyze_hedging_performance.py` script that:

- Loads the raw performance CSV files
- Aggregates results by strike type, hedging interval, and brokerage fee
- Generates summary statistics (mean MSE, standard deviation, etc.)
- Outputs `*_performance_summary.csv` files

**Running analysis**:

```bash
cd delta
python analyze_hedging_performance.py

cd delta_gamma
python analyze_hedging_performance.py

cd delta_vega
python analyze_delta_vega_performance.py
```

The delta-vega analysis script can also analyze both delta-vega results and optionally the original delta hedging results.

## Key Assumptions

- **Risk-free rate**: 6% (r = 0.06)
- **Strike prices**: 165, 170, 185, 190
- **Option type**: European call options
- **Pricing model**: Black-Scholes
- **Brokerage fees**: 0% or 2% of transaction value (depending on scenario)
- **Transaction costs** (Delta-Vega): 5% of traded value

## Dependencies

- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical computations
- `scipy` - Statistical functions (normal distribution)
- `refinitiv.dataplatform` - For fetching market data (used in `option_ric_tools.py` and `data_fetching.ipynb`)

## Results Interpretation

The Mean Squared Error (MSE) measures the hedging effectiveness:

- **Lower MSE** = Better hedging performance (smaller hedging errors)
- MSE is calculated as the average squared difference between:
  - Change in option position value
  - Change in replicating portfolio value (hedge)

The strategies are compared across different scenarios to understand:

- Impact of re-hedging frequency
- Effect of transaction costs
- Performance across different strike types
- Relative effectiveness of delta-only vs. multi-Greek hedging
