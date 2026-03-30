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


    

if __name__ == "__main__":
    calibrate_pruning()