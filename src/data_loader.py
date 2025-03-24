import os
import yfinance as yf
import pandas as pd

def get_sp500_data(start, end):
    # Create the directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    spy = yf.Ticker("SPY")
    data = spy.history(start=start, end=end)
    data.to_csv('../data/sp500_data.csv')
    return data['Close']

if __name__ == "__main__":
    data = get_sp500_data('2010-01-01', '2025-01-01')
    print("S&P 500 Data Collected!")
