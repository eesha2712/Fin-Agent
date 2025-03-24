import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: Each weight between 0% and 100%
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: Equal weight distribution
    initial_guess = num_assets * [1. / num_assets]
    
    # Minimize negative Sharpe Ratio (maximize reward/risk)
    def min_func(weights):
        ret, risk = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        return -ret / risk
    
    result = minimize(min_func, initial_guess, bounds=bounds, constraints=constraints)
    return result.x

if __name__ == "__main__":
    data = pd.read_csv('../data/sp500_data.csv', index_col=0, parse_dates=True)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    weights = optimize_portfolio(mean_returns, cov_matrix)
    print("Optimized Portfolio Weights:", weights)
