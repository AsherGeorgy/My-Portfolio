# Portfolio Optimization with Python

# Table of Contents

1. [Introduction](#introduction)
2. [Mean Variance Optimization](#mean-variance-optimization)
3. [Markowitz Portfolio Theory](#markowitz-portfolio-theory)
    - [Efficient Frontier](#efficient-frontier)
4. [Implementation with Python](#implementation-with-python)
    - [Monte Carlo Simulation](#monte-carlo-simulation)
    - [Optimization with CVXPY](#optimization-with-cvxpy)
    - [visualizaton](#visualization)
    - [Running the Program](#running-the-program)
    - [Outputs](#outputs)
    - [Analysis](#analysis)
    - [Conclusion](#conclusion)


# Introduction
This project performs comprehensive portfolio analysis, utilizing historical market data to assess the performance and optimize the allocation of assets. 
- Incorporates mean-variance optimization techniques, to strike a balance between risk and return while considering a specified minimum return constraint.
- Implements the principles of Markowitz Portfolio Theory by leveraging statistical measures to evaluate individual assets and construct efficient portfolios.
- Benchmarks portfolios against the S&P 500 index.
- Creates interactive visualizations to aid decision making. 

# Mean Variance Optimization
Mean-variance optimization is a fundamental concept in modern portfolio theory, pioneered by Harry Markowitz. It helps investors determine the biggest reward at a given level of risk or the least risk at a given level of return.     

In this project, the **cvxpy module** is used to find the optimal allocation of assets in a portfolio to minimize risk (as measured by volatility) at a target expected return. This enables investors to construct portfolios that offer the best trade-off between risk and target return based on historical market data.

**cvxpy** is a Python library for convex optimization that provides a simple and intuitive way to formulate and solve optimization problems. Convex optimization is particularly well-suited for mean-variance portfolio optimization because the problem formulation involves convex quadratic functions subject to linear constraints, which are readily solvable using convex optimization techniques.

# Markowitz Portfolio Theory

Markowitz Portfolio Theory revolutionized modern portfolio management by introducing the concept of diversification and optimizing portfolios for risk and return. He proved the existence of an efficient set of portfolios that optimize investor return for the amount of risk they are willing to accept. To do this, investments in multiple securities shouldn't be analyzed separately, but should be considered in a portfolio. Through the combination of securities with low correlation, investors can optimize their returns without assuming additional risk.

- **Diversification**: By holding a mix of assets that are not perfectly correlated, investors can achieve a more favorable risk-return profile for a given level of risk.

- **Risk and Return**: The goal of investors is to maximize returns while minimizing risk by combining assets in a way that their individual risks offset each other to some extent.

- **Covariance and Correlation**: Diversification is most effective when assets have low or negative correlations. Lower the correlation coefficient, the greater the diversification effect the stocks will have, i.e. the combined standard deviation decreases (the $(2w_1\sigma_1w_2\sigma_2\rho_{12})$ term in $(w_1\sigma_1 + w_2\sigma_2)^2$).

## **Efficient Frontier**
Graphical representation of optimal portfolios that offer the maximum expected return for a given level of risk or the minimum risk for a given level of expected return. The concept of the efficient frontier helps investors visualize the range of possible portfolios and select the one that aligns with their risk tolerance and return objectives.

**Example:** <br>
Assuming there are only two hypothetical companies in an economy, here is an example of the portfolio's efficient frontier constructed on Microsoft Excel:

![Efficient Frontier Example](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/Eff%20Frontier%20example.png)


This is precisely what Markowitz suggests: There is a set of efficient portfolios that can provide a higher expected rate of return for the same or even lower risk. This group of portfolios is called the Efficient Frontier.

- Its starting point is the minimum variance portfolio, the lowest risk an investor could bear. 
- Points below the efficient frontier represent inefficient portfolios, as for each, there exists an alternative portfolio with greater expected return for the same level of standard deviation.

# **Implementation with Python**

**Import necessary libraries and modules:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from fredapi import Fred
import cvxpy as cp
import plotly.graph_objs as go
```

<br>**Retrieve adjusted closing price data of the assets and benchmark index:** 
```python
def retrieve_data(assets, benchmark_index=None, no_of_years=None):
    """
    Retrieve historical adjusted closing prices for a given set of assets and a benchmark index from Yahoo Finance.

    Parameters:
    -----------
    assets : list of str
        List of asset symbols to retrieve data for. Each symbol should be a valid ticker symbol.
    benchmark_index : str, optional
        Symbol of the benchmark index to retrieve data for. Default is '^GSPC' (S&P 500).
    no_of_years : int, optional
        Number of years of historical data to retrieve. Default is 10 years.

    Returns:
    --------
    Tuple of pandas DataFrames:
        - adj_close: DataFrame containing adjusted closing prices for each asset.
        - benchmark_df: DataFrame containing adjusted closing prices for the benchmark index.
        - combined_df: DataFrame containing adjusted closing prices for assets and benchmark index combined.
        - benchmark_name: Name of the benchmark index.
    """

    no_of_years = no_of_years or 10
    benchmark_index = benchmark_index or '^GSPC'
    
    # Determine start_date and end_date based on no_of_years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(no_of_years*365.25))
    
    # Assets data
    adj_close = pd.DataFrame()
    for t in assets:
        adj_close[t] = yf.download(t, start_date, end_date)['Adj Close']
        if adj_close[t].empty:
            raise ValueError(f"No data available for asset: {t}")
    
    # Benchmark index data
    benchmark_df = pd.DataFrame()
    benchmark_df[benchmark_index] = yf.download(benchmark_index, start_date, end_date)['Adj Close']
    if benchmark_df.empty:
        raise ValueError(f"No data available for benchmark index: {benchmark_index}")
    
    # Retrieve long names of assets and benchmark index
    asset_names = []
    for t in assets:
        t = yf.Ticker(t)
        company_name = t.info['longName']
        asset_names.append(company_name)

    benchmark_ticker = yf.Ticker(benchmark_index)
    benchmark_name = benchmark_ticker.info['longName']

    # Print 
    print(f'\nThe following analysis is based on {no_of_years}Y daily adjusted closing price data from Yahoo Fnance.')
    print(f'\nTime period of analysis:    {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    print(f'Assets analysed:            {", ".join(asset_names)}')
    print(f'Index used as benchmark:    {benchmark_name}\n')
    
    # Merge benchmark_df with adj_close based on common index
    combined_df = pd.merge(adj_close, benchmark_df, left_index=True, right_index=True)
    
    return adj_close, benchmark_df,combined_df, benchmark_name
```
<br>**Retrieve risk free rate:** <br><br> 
**Fred API** retrieves risk free rate from Federal Reserve Economic Data (FRED) provided by the Federal Reserve Bank of St. Louis.   
```python
def retrieve_risk_free_rate(api_key=None):
    """
    Retrieve the risk-free rate.

    Parameters:
    ----------
        api_key (str, optional): API key for data source. Defaults to None.

    Returns:
    -------
        float: Risk-free rate.
    """
    api_key = api_key or '5dbe996305fb72e106205e372aa3Ofb8'
    fred = Fred(api_key=api_key)
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    return risk_free_rate
```
  "GS10" refers to the 10-year Treasury Constant Maturity Rate. This is the interest rate at which the U.S. government bonds with a maturity of approximately 10 years are issued (assumed to be risk free) <br><br>
**Calculate returns, covariance and correlation between each asset:**
```python
def return_stats(adj_close, benchmark_df, combined_df, assets, benchmark_index, no_of_years):
    """
    Calculate return statistics for assets and the benchmark index.

    Parameters:
    -----------
    adj_close : pandas.DataFrame
        DataFrame containing adjusted close prices of assets.
    benchmark_df : pandas.DataFrame
        DataFrame containing adjusted close prices of the benchmark index.
    combined_df : pandas.DataFrame
        DataFrame containing adjusted close prices of assets and the benchmark index.
    assets : list of str
        List of asset tickers.
    benchmark_index : str
        Ticker symbol of the benchmark index.
    no_of_years : int
        Number of years of historical data used in the analysis.

    Returns:
    --------
    returns_assets : pandas.DataFrame
        DataFrame containing daily returns of assets.
    returns_assets_ann : pandas.DataFrame
        DataFrame containing annualized returns of assets.
    returns_assets_cov : pandas.DataFrame
        DataFrame containing covariance matrix of asset returns.
    returns_benchmark : pandas.DataFrame
        DataFrame containing daily returns of the benchmark index.
    """
    # Calculate simple returns, covariance and correlation of all including benchmark
    returns_all = combined_df.pct_change().dropna()
    returns_all_ann = returns_all * 250
    returns_all_cov = returns_all.cov() * 250
    returns_all_corr = returns_all.corr()
    
    # Calculate simple returns, covariance and correlation of just the assets
    returns_assets = adj_close.pct_change().dropna()
    returns_assets_ann = returns_assets * 250
    returns_assets_cov = returns_assets.cov() * 250
    
    # Calculate simple returns of just the benchmark
    returns_benchmark = benchmark_df.pct_change().dropna()
    
    # Print Returns, Volatility and Correlation matrix of the assets and benchmark index
    tickers = assets + [benchmark_index]
    print(f'\n\nIndividual Asset Analysis: \n\nAnnualized {no_of_years} year Total Returns (Daily):')
    for ticker in tickers:
        print(f'  {ticker}: {returns_all_ann[ticker].mean()*100:.2f}%')
    print()

    print(f'Annual Volatility ({no_of_years}Y):')
    for ticker in tickers:
        print(f'  {ticker}: {(returns_all[ticker].std() * 250 ** 0.5)*100:.2f}%')
    print()

    print(f'Correlation matrix: \n{np.round(returns_all_corr,3)}\n')
    
    return returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark
```

**Calculate Portfolio return, volatility and sharpe ratio:**
```python
def portfolio_stats(weights, returns_assets, cov_assets, risk_free_rate):
    """
    Calculate portfolio statistics including return, volatility, and Sharpe ratio.

    Parameters:
    -----------
    weights : numpy.ndarray
        Array containing the weights of each asset in the portfolio.
    returns_assets : pandas.DataFrame
        DataFrame containing the annualized returns of individual assets.
    cov_assets : pandas.DataFrame
        DataFrame containing the covariance matrix of returns of individual assets.
    risk_free_rate : float
        The risk-free rate of return.
    Returns:
    --------
    tuple
        A tuple containing the calculated portfolio statistics:
        - portfolio_return : float
            The expected annual return of the portfolio.
        - portfolio_volatility : float
            The annualized volatility (standard deviation) of the portfolio.
        - sharpe_ratio : float
            The Sharpe ratio of the portfolio, calculated as (portfolio_return - risk_free_rate) / portfolio_volatility.
    """
    # Portfolio Return
    portfolio_return = np.dot(weights, returns_assets.mean())
    
    # Portfolio Volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_assets, weights)))
    
    # Portfolio Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio
```

## Monte Carlo Simulation
To plot the Efficient frontier, multiple iterations of the portfolio with varying weights are required. A 'for' loop combined with NumPy's 'random' module can be used to achieve this. 

>The **np.random.random()** is used to obtain random floats counting equal to the count of tickers. Each float is then divided by the numpy sum of the floats to get the weights, such that the sum of weights = 1. The logic is $\frac{a}{a+b} + \frac{b}{a+b} = 1$.

```python
def eff_frontier(assets, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations=1000):
    """
    Generate the efficient frontier of a portfolio using Monte Carlo simulation.

    Parameters:
    -----------
    assets : list of str
        List of asset tickers or names.
    returns_assets_ann : pandas DataFrame
        DataFrame containing annualized returns of individual assets.
    returns_assets_cov : pandas DataFrame
        DataFrame containing covariance matrix of asset returns.
    risk_free_rate : float
        Risk-free rate of return.
    no_of_iterations : int, optional
        Number of iterations for Monte Carlo simulation. Default is 1000.

    Returns:
    --------
    pfolio_volatility : numpy array
        Array containing portfolio volatilities for each iteration.
    pfolio_return : numpy array
        Array containing portfolio returns for each iteration.
    weights : numpy array
        Array containing portfolio weights for each iteration.
    sharpe_ratios : numpy array
        Array containing Sharpe ratios for each portfolio iteration.
    """
    # Initialize list to append portfolio returns, volatility and weights from each iteration.    
    pfolio_return = []
    pfolio_volatility = []
    weights_list = []
    
    
    # Generate random weights and calculate portfolio returns and volatility based on them. 
    for i in range (no_of_iterations):
        no_of_tickers = len(assets)
        random_floats = np.random.random(no_of_tickers)
        weights = random_floats/np.sum(random_floats)
        
        weights_list.append(weights)
        pfolio_return.append(np.dot(returns_assets_ann.mean(), weights))
        pfolio_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns_assets_cov, weights))))
        
    # Convert each lists to numpy arrays 
    weights = np.array(weights_list)
    pfolio_return = np.array(pfolio_return)*100
    pfolio_volatility = np.array(pfolio_volatility)*100
    sharpe_ratios = (pfolio_return - risk_free_rate) / pfolio_volatility
    
    return pfolio_volatility, pfolio_return, weights, sharpe_ratios
```

From the simulated data, portfolios with the highest return, highest sharpe ratio and lowest volatility can be identified along with with their corresponding weights (see visualize_analyses() function).

## Optimization with CVXPY
This function uses convex optimization to find the optimal portfolio allocation that minimizes risk while achieving a specified minimum return. It formulates the optimization problem as follows:  

Objective - Minimize volatility:   $w^T * Cov * w$    

Constraint 1 - Target Return:   $w^T$ * `mean_returns (ret)` >= `target returns (min_return)`  

Constraint 2 - Sum of weights:   $sum(w) == 1$
```python
def opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, risk_free_rate, min_return):
    """
    Optimize portfolio allocation using Mean-Variance Optimization (MVO) with convex optimization (cvxpy).

    Parameters:
    -----------
        returns_assets_ann (pd.DataFrame): Annualized returns of individual assets.
        returns_assets_cov (pd.DataFrame): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate used in the optimization.
        min_return (float): Target minimum return constraint for the portfolio.

    Returns:
    --------
        numpy.array: Optimal weights for the portfolio allocation.
    """
    # Initialize variables
    n = len(returns_assets_ann.columns)     # initialize variable for the number of assets (counted as the number of columns in the dataframe)
    w = cp.Variable(n)     # cp placeholder variable for n number of portfolio weights
    
    # Calculate expected Return and Risk
    ret = returns_assets_ann.mean().values @ w     # Expected return is the dot product of weights and mean returns of the individual assets
    risk = cp.quad_form(w, returns_assets_cov.values)     # calculated in quadratic form
    
    # Define the objective
    objective = cp.Minimize(risk)     # Minimize portfolio risk using Minimize function from cp
    
    # Define the constraints
    constraints = [
        cp.sum(w) == 1,     # Sum of weights equals 1 (fully invested)
        ret >= min_return     # Target minimum return constraint
    ]
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)     # cp 'Problem' object encapsulates the optimization problem, with defined objectives and constraints
    prob.solve()                                  # solves the problem using an appropriate solver selected based on problem structure and constraints
    
    # Retrieve the optimized weights from variable w
    optimal_weights = w.value
    
    return optimal_weights 
```
```python
def opt_portfolio_results(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate, assets, min_return, benchmark_index):
        """
    Calculate and display the results of mean-variance portfolio optimization.

    Parameters:
    -----------
    optimal_weights : numpy.ndarray
        Array containing the optimal weights for the portfolio allocation.
    returns_assets_ann : pandas.DataFrame
        DataFrame containing the annualized returns of individual assets.
    returns_assets_cov : pandas.DataFrame
        DataFrame containing the covariance matrix of returns for individual assets.
    risk_free_rate : float
        The risk-free rate used in the optimization process.
    assets : list of str
        List of asset symbols or names.
    min_return : float
        The target minimum return specified for the portfolio optimization.
    benchmark_index : str
        Symbol or name of the benchmark index against which the portfolio is compared.

    Returns:
    --------
    None
    """
    optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio = portfolio_stats(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate)
    
    print()
    print(f'\nPortfolio Analysis: \n\nA. Mean-Variance Optimization (cvxpy):\n\nPortfolio optimized to minimize volatility while achieving a target return of {min_return*100}% (subject to constraints): \n  1. Weights:')
    for ticker, weight in zip(assets, optimal_weights):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Expected Annual Return: {optimal_portfolio_return*100:.2f}%\n  3. Expected Volatility: {optimal_portfolio_volatility*100:.2f}%\n  4. Sharpe Ratio: {optimal_sharpe_ratio:.4f}\n')
```


## Visualization
The **matplotlib** and **plotly** libraries are used to create interactive visualizations. 
```python
def visualize_analyses(pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, assets, benchmark_index):
    """
    Visualizes the results of portfolio analyses including the efficient frontier, identified portfolios, and their relative daily return movements.

    Parameters:
    -----------
    pfolio_volatility : array-like
        Array containing the volatilities of the portfolios on the efficient frontier.
    pfolio_return : array-like
        Array containing the returns of the portfolios on the efficient frontier.
    weights : array-like
        Array containing the weights of the portfolios on the efficient frontier.
    sharpe_ratios : array-like
        Array containing the Sharpe ratios of the portfolios on the efficient frontier.
    returns_assets : pandas DataFrame
        DataFrame containing the daily returns of the individual assets.
    optimal_weights : array-like
        Array containing the optimized weights of the portfolio.
    returns_benchmark : pandas DataFrame
        DataFrame containing the daily returns of the benchmark index.
    benchmark_name : str
        Name of the benchmark index.
    assets : list of str
        List of asset tickers.
    benchmark_index : str
        Ticker symbol of the benchmark index.
    Returns:
    --------
    None

    Displays visualizations including:
    - Efficient frontier curve showing the trade-off between risk (volatility) and return.
    - Daily return movements of identified portfolios against the benchmark index.
    - Textual output summarizing the characteristics of identified portfolios.
    """
    # Identify relevant portfolios (Return and volatility are returned in percentages; see eff_frontier()) 
    
    # 1. Identify minimum volatility portfolio
    min_volatility_idx = np.argmin(pfolio_volatility)     # Returns the indices of the minimum values
    min_volatility = pfolio_volatility[min_volatility_idx]
    return_at_min_volatility = pfolio_return[min_volatility_idx]
    weights_at_min_volatility = weights[min_volatility_idx]
    sharpe_ratio_at_min_volatility = sharpe_ratios[min_volatility_idx]
    
    # 2. Identify maximum return portfolio
    max_return_idx = np.argmax(pfolio_return)     # Returns the indices of the maximum values
    max_return = pfolio_return[max_return_idx]
    vol_at_max_return = pfolio_volatility[max_return_idx]
    weights_at_max_return = weights[max_return_idx]
    sharpe_ratio_at_max_return = sharpe_ratios[max_return_idx]
    
    # 3. Identify maximum sharpe ratio portfolio
    max_sharpe_idx = np.argmax(sharpe_ratios)     # Returns the indices of the maximum values
    max_sharpe = sharpe_ratios[max_sharpe_idx]
    return_at_max_sharpe = pfolio_return[max_sharpe_idx]
    volatility_at_max_sharpe = pfolio_volatility[max_sharpe_idx]
    weights_at_max_sharpe = weights[max_sharpe_idx]
    
    
    # Calculate daily returns of each of the above portfolios to plot against benchmark_index:
    
    # 1. Use the weights to calculate daily returns of maximum sharpe ratio portfolio
    pfolio_returns_at_max_sharpe_weights = np.dot(returns_assets, weights_at_max_sharpe)   
    
    # 2. Use the weights to calculate daily returns of minimum volatility portfolio
    pfolio_returns_at_min_vol_weights = np.dot(returns_assets, weights_at_min_volatility)
    
    # 3. Use the weights to calculate daily returns of maximum return portfolio
    pfolio_returns_at_max_ret_weights = np.dot(returns_assets, weights_at_max_return)
    
    # 4. Use optimal weights to calculate daily returns of the optimized portfolio
    pfolio_returns_at_optimal_weights = np.dot(returns_assets, optimal_weights)
    
    # 5. Merge each of the above numpy arrays with the returns_benchmark dataframe 
    cumprod_df = returns_benchmark.copy()
    cumprod_df.rename(columns={benchmark_index:benchmark_name}, inplace=True)              # Rename column from ticker to full name
    cumprod_df['Maximum Sharpe ratio Porfolio']  = pfolio_returns_at_max_sharpe_weights
    cumprod_df['Minimum Volatility Porfolio']  = pfolio_returns_at_min_vol_weights
    cumprod_df['Maximum Return Porfolio']  = pfolio_returns_at_max_ret_weights
    cumprod_df['Optimized Porfolio']  = pfolio_returns_at_optimal_weights
    
    # 6. Calculate cumulative product of the dataframe in order to compare relative movements
    cumprod_df = (1 + cumprod_df).cumprod() - 1
    
    
    # Plot Relative return movements of Optimal Portfolio Vs Benchmark Index
    cumprod_df[[benchmark_name, 'Minimum Volatility Porfolio']].plot(figsize=(10,6))
    plt.title(f'\nComparing Relative Daily Return Movements: Optimized Portfolio Vs {benchmark_name}\n')
    plt.show() 
    print()
    
    
    # Plot the efficient frontier curve:
    print('\n\n\nB. Markowitz Portfolio Analysis:\n')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pfolio_volatility, y=pfolio_return, mode='markers', name='Portfolios'))
    fig.add_trace(go.Scatter(x=[min_volatility], y=[return_at_min_volatility], mode='markers', name='Min Volatility', marker=dict(color='green', size=10)))             # Marker for Minimium Volatility Portfolio
    fig.add_trace(go.Scatter(x=[vol_at_max_return], y=[max_return], mode='markers', name='Max Return', marker=dict(color='blue', size=10)))                             # Marker for Maximum Return Portfolio
    fig.add_trace(go.Scatter(x=[volatility_at_max_sharpe], y=[return_at_max_sharpe], mode='markers', name='Max Sharpe Ratio', marker=dict(color='red', size=10)))       # Marker for Maximum Sharpe Ratio
    
    fig.update_layout(title='Interactive Efficient Frontier Visualization',
                      xaxis_title='Standard Deviation (%)',
                      yaxis_title='Portfolio Return (%)')
    fig.show()
    

    # Print each portfolio identified and plot their daily returns against benchmark index 
    # 1. Minimum volatility portfolio
    print(f'\nEfficient Frontier Portfolios:\n\n(a) Minimum volatility portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_min_volatility):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Portfolio Return: {return_at_min_volatility:.2f}%\n  3. Portfolio Volatility: {min_volatility:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_min_volatility:.4f}\n')

    # 2. Maximum return portfolio
    print(f'(b) Maximum return portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_max_return):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Expected Annual Return: {max_return:.2f}%\n  3. Expected Volatility: {vol_at_max_return:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_max_return:.4f}\n')
    
    # 3. Maximum sharpe ratio portfolio
    print(f'(c) Maximum Sharpe Ratio portfolio: \n  1. Weights:')
    for ticker, weight in zip(assets, weights_at_max_sharpe):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Portfolio Return: {return_at_max_sharpe:.2f}%\n  3. Portfolio Volatility: {volatility_at_max_sharpe:.2f}%\n  4. Sharpe Ratio:{max_sharpe:.4f}\n')
    
    
    # Plot relative daily return movements
    # 1. Minimum volatility portfolio
    cumprod_df[[benchmark_name, 'Minimum Volatility Porfolio']].plot(figsize=(10,6))
    plt.title(f'\nComparing Relative Daily Return Movements: Minimum Volatility Portfolio Vs {benchmark_name}\n')
    plt.show() 
    print()  
    
    # 2. Maximum return portfolio
    cumprod_df[[benchmark_name, 'Maximum Return Porfolio']].plot(figsize=(10,6))
    plt.title(f'Comparing Relative Daily Return Movements: Maximum Return Porfolio Vs {benchmark_name}\n')
    plt.show()
    print() 
    
    # 3. Maximum sharpe ratio portfolio
    cumprod_df[[benchmark_name, 'Maximum Sharpe ratio Porfolio']].plot(figsize=(10,6))
    plt.title(f'Comparing Relative Daily Return Movements: Maximum Sharpe Ratio Portfolio Vs {benchmark_name}\n')
    plt.show()
    
    
    return 
```

## Running the Program
The program is run for a portfolio of Apple and Walmart stocks, for a target return of 10% and a timeframe of 10 years. The result is benchmarked against the S&P500 index.
```python
def main():
    """
    Main function to orchestrate the portfolio analysis process.

    This function retrieves historical market data, calculates various statistics,
    performs mean-variance optimization, visualizes the results, and prints summaries.

    Returns:
    -------
    None
        The function doesn't return any value explicitly but performs the analysis
        and prints the results.
    """

    # Inputs
    assets = ['WMT', 'AAPL']                        # Assets to be analyzed. Must be provided as a list of strings
    benchmark_index = '^GSPC'                       # The index against which the assets and porfolio is to be compared with. Default is the S&P500 index
    no_of_years = 10                                # The number of years of historical data on which the analysis is to be conducted. Default is 10 years
    
    api_key = '5dbe996305fb72e106205e372aa3Ofb8'    # API key for Fred API
    min_return = 0.20                               # Minimum return constraint for optimization with cvxpy
    
    
    # Processes and Outputs:
    
    # 1. Retrieve historical market data
    adj_close, benchmark_df, combined_df, benchmark_name = retrieve_data(assets, benchmark_index, no_of_years)
    
    # 2. Calculate return and risk statistics
    returns_assets, returns_assets_ann, returns_assets_cov, returns_benchmark = return_stats(adj_close, benchmark_df, combined_df, assets, benchmark_index, no_of_years)
    
    # 3. Retrieve the risk-free rate
    risk_free_rate = retrieve_risk_free_rate()
    
    # 4. Perform efficient frontier analysis
    pfolio_volatility, pfolio_return, weights, sharpe_ratios = eff_frontier(assets, returns_assets_ann, returns_assets_cov, risk_free_rate, no_of_iterations=1000)
    
    # 5. Optimize portfolio using mean-variance optimization
    optimal_weights = opt_portfolio_cvxpy(returns_assets_ann, returns_assets_cov, risk_free_rate, min_return)
    
    # 6. Print optimized portfolio results
    opt_portfolio_results(optimal_weights, returns_assets_ann, returns_assets_cov, risk_free_rate, assets, min_return, benchmark_index)
    
    # 7. Visualize analysis results
    visualize_analyses(pfolio_volatility, pfolio_return, weights, sharpe_ratios, returns_assets, optimal_weights, returns_benchmark, benchmark_name, assets, benchmark_index)

if __name__ == "__main__":
    main()
```
## Outputs
**The above code received the following outputs:**
```
The following analysis is based on 10Y daily adjusted closing price data from Yahoo Fnance.

Time period of analysis:    2014-04-02 to 2024-04-01
Assets analysed:            Walmart Inc., Apple Inc.
Index used as benchmark:    S&P 500
Risk free rate used:        4.21%


Individual Asset Analysis: 

Annualized 10 year Total Returns (Daily):
  WMT: 12.69%
  AAPL: 26.94%
  ^GSPC: 11.73%

Annual Volatility (10Y):
  WMT: 20.72%
  AAPL: 28.24%
  ^GSPC: 17.66%

Correlation matrix: 
         WMT   AAPL  ^GSPC
WMT    1.000  0.323  0.442
AAPL   0.323  1.000  0.748
^GSPC  0.442  0.748  1.000



Portfolio Analysis: 

A. Mean-Variance Optimization (cvxpy):

Portfolio optimized to minimize volatility while achieving a target return of 20.0% (subject to constraints): 
  1. Weights:
      WMT: 0.487
      AAPL: 0.513
  2. Expected Annual Return: 20.00%
  3. Expected Volatility: 20.15%
  4. Sharpe Ratio: 0.7835
```
![Optimized](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/output_10_2.png)

```
B. Markowitz Portfolio Analysis:
```
![Interactive Eff Frontier](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/output_10_5.png)
```
Efficient Frontier Portfolios:

(a) Minimum volatility portfolio: 
  1. Weights:
      WMT: 0.716
      AAPL: 0.284
  2. Portfolio Return: 16.73%
  3. Portfolio Volatility: 19.01%
  4. Sharpe Ratio:0.8780

(b) Maximum return portfolio: 
  1. Weights:
      WMT: 0.000
      AAPL: 1.000
  2. Expected Annual Return: 26.93%
  3. Expected Volatility: 28.23%
  4. Sharpe Ratio:0.9527

(c) Maximum Sharpe Ratio portfolio: 
  1. Weights:
      WMT: 0.353
      AAPL: 0.647
  2. Portfolio Return: 21.91%
  3. Portfolio Volatility: 21.77%
  4. Sharpe Ratio:1.0048
```

![Min Volatility](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/output_10_7.png)
![Max Return](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/output_10_9.png)
![Max Sharpe](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/output_10_11.png)


## Analysis

The following insights can be drawn based on the above analysis of Apple (AAPL) and Walmart (WMT) stocks:<br>
- AAPL vs. WMT:
   - AAPL offers higher potential returns but also comes with higher volatility, indicating greater risk.
   - WMT offers more moderate returns with lower volatility, providing stability to the portfolio.
- Portfolio Allocation:
   - Including both AAPL and WMT in a portfolio allows investors to balance risk and return based on their risk tolerance and investment objectives.
   - Investors seeking lower volatility may prefer a portfolio with a higher allocation to WMT, while those aiming for higher returns may lean towards AAPL or a combination of both assets based on their risk-return preferences.
- Diversification:
   - Combining AAPL and WMT in a portfolio provides diversification benefits, as these assets may have different performance drivers and exhibit varying levels of correlation with each other and the broader market (S&P 500).
   - The portfolio analysis highlights the importance of diversification in managing risk and optimizing returns, as seen in the efficient frontier portfolios.

## Conclusion
The analysis underscores the fundamental risk-return tradeoff in investing. Portfolios with higher expected returns typically comes with increased volatility.   
It's essential for investors to assess their risk tolerance, investment objectives, and time horizon before making investment decisions. Diversification, thorough research, and understanding individual company fundamentals are crucial aspects of constructing a well-balanced investment portfolio. 
