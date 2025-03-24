from src.data_loader import get_sp500_data
from src.portfolio import optimize_portfolio
from src.evaluation import evaluate_portfolio
import numpy as np

if __name__ == "__main__":
    # Load data
    data = get_sp500_data('2010-01-01', '2025-01-01')
    
    # Calculate daily returns and convert to DataFrame
    returns = data.pct_change().dropna().to_frame()

    # Compute mean return and variance
    mean_returns = returns.mean()
    cov_matrix = returns.var()

    # Calculate Sharpe ratio
    sharpe_ratio = mean_returns[0] / np.sqrt(cov_matrix[0])
    print(f"Expected Return: {mean_returns[0]:.4f}, Risk (Volatility): {np.sqrt(cov_matrix[0]):.4f}, Sharpe Ratio: {sharpe_ratio:.4f}")

    # Portfolio optimization (weight is always 1 for single asset)
    weights = np.array([1.0])
    print("Optimized Portfolio Weights:", weights)

    # Evaluate performance
    evaluate_portfolio(weights, returns)
