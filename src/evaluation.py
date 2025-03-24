import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

def evaluate_portfolio(weights, returns):
    if returns.shape[1] == 1:
        # Single asset case
        portfolio_returns = returns.iloc[:, 0] * weights[0]
    else:
        portfolio_returns = (returns * weights).sum(axis=1)
    
    cumulative_return = np.exp(np.log1p(portfolio_returns).sum()) - 1
    print(f"Cumulative Return: {cumulative_return:.4f}")


if __name__ == "__main__":
    data = pd.read_csv('../data/sp500_data.csv', index_col=0, parse_dates=True)
    weights = [0.25, 0.25, 0.25, 0.25]
    evaluate_portfolio(weights, data)
