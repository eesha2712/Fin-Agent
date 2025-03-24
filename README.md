
# S&P 500 AI-Powered Portfolio Optimizer

## Overview
This app allows users to input their portfolio of S&P 500 stocks, optimize the portfolio using Modern Portfolio Theory (MPT), and dynamically adjust the portfolio weights using a reinforcement learning agent. The goal is to maximize returns while minimizing risk (volatility), providing a long-term investment strategy.

## Features
- **Portfolio Input:** Enter your S&P 500 stock tickers and their respective weights.
- **Data Fetching:** Automatically fetches historical stock data from Yahoo Finance using `yfinance`.
- **Portfolio Optimization:** Uses Modern Portfolio Theory (MPT) to calculate optimal portfolio weights for maximizing returns and minimizing risk.
- **Reinforcement Learning:** Implements a simple reinforcement learning agent for dynamic rebalancing based on market conditions (using a Deep Q-Network).
- **Performance Evaluation:** Visualizes portfolio performance with cumulative returns over time.
- **Real-time Updates:** Rebalances the portfolio and displays updated predictions.

## Requirements
Before running the app, ensure you have the following dependencies installed:

- **Python 3.x**
- **Required Libraries:**
  - `streamlit`
  - `yfinance`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`

You can install the required libraries using the following:

```bash
pip install streamlit yfinance numpy pandas matplotlib scikit-learn tensorflow
