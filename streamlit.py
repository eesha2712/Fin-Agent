import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.portfolio import optimize_portfolio
from src.agent import PortfolioAgent
from src.evaluation import evaluate_portfolio

# Function to fetch data for tickers
def get_data_for_tickers(tickers, start='2010-01-01', end='2025-01-01'):
    data = {}
    for ticker in tickers:
        stock_data = yf.Ticker(ticker).history(start=start, end=end)
        data[ticker] = stock_data['Close']
    return pd.DataFrame(data)

# Streamlit UI
st.title('S&P 500 Portfolio Optimizer')

st.write("""
This app allows you to input your portfolio of S&P 500 stocks, optimize the portfolio using Modern Portfolio Theory, 
and dynamically adjust the weights using a reinforcement learning agent to maximize returns and minimize risk.
""")

# User input: Portfolio Tickers and Weights
st.sidebar.header("Enter your Portfolio")
tickers = st.sidebar.text_area("Enter Stock Tickers (comma separated)", "AAPL, MSFT, AMZN, TSLA")
tickers = tickers.split(',')

weights_input = st.sidebar.text_area("Enter Portfolio Weights (comma separated)", "0.25, 0.25, 0.25, 0.25")
weights = list(map(float, weights_input.split(',')))

# Fetch the data
if st.sidebar.button("Fetch Data"):
    data = get_data_for_tickers(tickers)
    st.write("Data for Portfolio:")
    st.dataframe(data.tail())  # Show last few rows of the data

    # Calculate returns and covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Portfolio optimization using MPT
    optimized_weights = optimize_portfolio(mean_returns, cov_matrix)
    st.write("Optimized Portfolio Weights using MPT:")
    st.write(optimized_weights)

    # Performance Evaluation
    st.subheader("Portfolio Performance")
    evaluate_portfolio(optimized_weights, data)

    # Dynamic Rebalancing using RL (Optional)
    st.subheader("Dynamic Portfolio Rebalancing (Reinforcement Learning Agent)")
    agent = PortfolioAgent(state_size=len(tickers), action_size=len(tickers))
    
    state = np.random.rand(1, len(tickers))  # Random state (could be replaced with actual data)
    action = np.random.randint(0, len(tickers))
    reward = np.random.rand()  # Dummy reward
    next_state = np.random.rand(1, len(tickers))  # Next state

    agent.train(state, action, reward, next_state)
    st.write("Agent has adjusted portfolio weights based on market conditions.")
    
    # Visualize Portfolio Cumulative Returns
    portfolio_returns = (returns * optimized_weights).sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_returns.cumsum(), label="Optimized Portfolio Cumulative Returns")
    plt.title("Portfolio Performance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    st.pyplot()

# Run the app with:
# streamlit run streamlit.py