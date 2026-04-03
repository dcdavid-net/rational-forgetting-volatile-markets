import pandas as pd
import numpy as np
from math import log

def calibrate_pruning(input_filename="../data/sp500_historical_returns.csv", decay_rate=0.5, n_iterations=10000):
    # Implement bootstrapping for statistical robustness
    df = pd.read_csv(input_filename, index_col='Date', parse_dates=True)

    # MSCI's RiskMetrics Technical Document.pdf
    riskmetrics_lambda_daily = 0.94 # decay factor from RiskMetrics / JP Morgan
    alpha = 1 - riskmetrics_lambda_daily # 
    lagged_returns = df['Log_Return'].shift(1) # ensure we're only looking in the past
    lagged_squared_returns = lagged_returns ** 2 # expected value of the squared returns are variances
    df['EWMA_variance'] = lagged_squared_returns.ewm(alpha=alpha, adjust=False).mean() # equation 5.3 $\lambda\sigma_{1,t|t-1}^{2}+(1-\lambda)r_{1,t}^{2}$ 
    df['EWMA_volatility'] = np.sqrt(df['EWMA_variance']) # equation 5.4 basically saying EWMA_vol is sqrt(EWMA_variance)
    df.dropna(inplace=True)
    df = df[df['EWMA_volatility'] > 0]

    print(df.head())

    # Z = (val - mean) / (std). Assuming zero mean, Z = val / std
    df['Z_Score'] = df['Log_Return'] / df['EWMA_volatility'] 

    # agent has to be able to remember majority of extreme negative events
    # Philippe Jorion’s Value at Risk / McNeil Extreme Value Theory for Risk Managers
    # Herbert Simon Bounded Rationality ('satisficing')
    # this needs to be 95th percentile. Bootstrap to be robust
    df_reset = df.reset_index()
    extreme_negative_events = df_reset[df_reset['Z_Score'] < -2.0].copy() 
    event_indices = extreme_negative_events.index.to_numpy()
    gaps = np.diff(event_indices)

    print(f'\nNumber of extreme events found: {len(extreme_negative_events)}')
    print(f'Theoretical 100% max gap: {gaps.max()} days')

    max_gap = gaps.max()
    max_gap_idx = gaps.argmax()
    
    start_event_date = extreme_negative_events.iloc[max_gap_idx]['Date'].date()
    end_event_date = extreme_negative_events.iloc[max_gap_idx + 1]['Date'].date()

    print(f'Longest extreme negative event range: {start_event_date} to {end_event_date}')

    # Ok now let's bootstrap to get a robust "95th percentile"
    np.random.seed(42)
    bootstrapped_95th_percentiles = []

    for _ in range(n_iterations):
        random_gap = np.random.choice(gaps, size=len(gaps), replace=True)
        p95 = np.percentile(random_gap, 95)
        bootstrapped_95th_percentiles.append(p95)
    bootstrapped_95th_percentiles = np.array(bootstrapped_95th_percentiles)
    print(f'Bootstrapped 95th percentiles: {bootstrapped_95th_percentiles}')

    # need upper 95% CI so we remember vast majority of normal periods
    # ci_lower = np.percentile(bootstrapped_95th_percentiles, 2.5)
    target_gap = np.percentile(bootstrapped_95th_percentiles, 97.5)
    print(f'Target gap 95% upper CI: {target_gap}')

    tau_prune = -decay_rate * log(target_gap)
    print(tau_prune)
    return tau_prune

if __name__ == "__main__":
    calibrate_pruning()