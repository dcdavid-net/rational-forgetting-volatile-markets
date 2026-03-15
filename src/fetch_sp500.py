import yfinance as yf
import pandas as pd
import numpy as np

def fetch_sp500(output_filename="../data/sp500_historical_returns.csv"):
    sp500 = yf.download('^GSPC', period='max', interval='1d')
    sp500.columns = sp500.columns.get_level_values(0)
    df = sp500[['Close']].copy()
    df.dropna(inplace=True) # clean up any days with no price
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True) # drop the first row
    df.to_csv(output_filename)
    return df

if __name__ == "__main__":
    fetch_sp500()