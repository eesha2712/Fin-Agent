
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

`pip install streamlit yfinance numpy pandas matplotlib scikit-learn tensorflow`

## How to Use the App
### Enter Portfolio Information:

In the sidebar, input the stock tickers of the companies in your portfolio (comma-separated).

Enter the corresponding portfolio weights (comma-separated) to allocate among your selected stocks.

### Fetch Data:

Click the "Fetch Data" button to retrieve historical stock data for the selected tickers using Yahoo Finance.

### Portfolio Optimization:

The app will display the optimized portfolio weights based on Modern Portfolio Theory (MPT) and visualize the portfolio's cumulative returns over time.

### Reinforcement Learning Agent:

The app simulates a reinforcement learning agent that dynamically adjusts the portfolio weights based on market conditions.
