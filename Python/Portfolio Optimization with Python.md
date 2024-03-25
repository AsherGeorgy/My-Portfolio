# Markowitz Portfolio Optimization with Python

# Table of Contents

1. [Introduction](#introduction)
2. [Markowitz Portfolio Theory](#markowitz-portfolio-theory)
    - [Efficient Frontier](#efficient-frontier)
3. [Portfolio Optimization with Python](#portfolio-optimization-with-python)
    - [Risk, Return, and Sharpe Ratio](#risk-return-and-sharpe-ratio)
    - [Monte Carlo Efficient Frontier Simulation](#monte-carlo-efficient-frontier-simulation)
    - [Optimization with Scipy](#optimization-with-scipy)
    - [Running the Optimization](#running-the-optimization)
    - [Outputs](#outputs)
    - [Analysis](#analysis)
    - [Conclusion](#conclusion)


# Introduction
This project implements the principles of Markowitz Portfolio Theory using Python. Markowitz Portfolio Theory, proposed by Harry Markowitz, revolutionized modern portfolio management by introducing the concept of diversification and optimizing portfolios for risk and return.

# Markowitz Portfolio Theory

Markowitz proved the existence of an efficient set of portfolios that optimize investor return for the amount of risk they are willing to accept. To do this, investments in multiple securities shouldn't be analyzed separately, but should be considered in a portfolio. Through the combination of securities with low correlation, investors can optimize their returns without assuming additional risk.

Markowitz assumes investors are rational and risk averse, so they are interested in earning higher returns and prefer avoiding additional risk. This leads to the conclusion that for any level of risk, investors will be interested only in the portfolio with the highest expected return.

Key points:
- **Diversification**: By holding a mix of assets that are not perfectly correlated, investors can achieve a more favorable risk-return profile for a given level of risk.

- **Risk and Return**: The goal of investors is to maximize returns while minimizing risk by combining assets in a way that their individual risks offset each other to some extent.

- **Covariance and Correlation**: Diversification is most effective when assets have low or negative correlations. Lower the correlation coefficient, the greater the diversification effect the stocks will have, i.e. the combined standard deviation decreases (the $(2w_1\sigma_1w_2\sigma_2\rho_{12})$ term in $(w_1\sigma_1 + w_2\sigma_2)^2$).

- **Portfolio Optimization**: The process of mathematically determining the best mix of assets to achieve the desired level of return with the least amount of risk. It involves using quantitative techniques, such as mathematical optimization models, to allocate capital across different assets. The goal is to construct a portfolio on the efficient frontier.

## **Efficient Frontier**
Graphical representation of optimal portfolios that offer the maximum expected return for a given level of risk or the minimum risk for a given level of expected return. The concept of the efficient frontier helps investors visualize the range of possible portfolios and select the one that aligns with their risk tolerance and return objectives.

**Example:** <br>
Assuming there are only two hypothetical companies in an economy, here is an example of the portfolio's efficient frontier constructed on Microsoft Excel:

![Efficient Frontier Example](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/Eff%20Frontier%20example.png)


This is precisely what Markowitz suggests: There is a set of efficient portfolios that can provide a higher expected rate of return for the same or even lower risk. This group of portfolios is called the Efficient Frontier.

- Its starting point is the minimum variance portfolio, the lowest risk an investor could bear. 
- Points below the efficient frontier represent inefficient portfolios, as for each, there exists an alternative portfolio with greater expected return for the same level of standard deviation.

# **Portfolio Optimization with Python**

## Risk, Return and Sharpe Ratio

**Import necessary libraries and modules:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
```

**Retrieve stock price data:** 
```python
def retrieve_data(tickers, start_date=None, end_date=None, no_of_years=None, interval='1d'):
    """
    Retrieve historical stock price data for the specified tickers.

    Args:
        tickers (list): List of ticker symbols for the stocks.
        start_date (str or datetime, optional): Start date for data retrieval. Defaults to None.
        end_date (str or datetime, optional): End date for data retrieval. Defaults to None.
        no_of_years (int, optional): Number of years of data to retrieve if start_date and end_date are not provided. Defaults to None.
        interval (str, optional): Interval for data (e.g., '1d' for daily). Defaults to '1d'.

    Returns:
        pandas.DataFrame: DataFrame containing the adjusted closing prices of the specified tickers.
        list: List of ticker symbols used in the analysis.
    """
    adj_close = pd.DataFrame()
    if start_date is None and end_date is None and no_of_years is None:
        start_date = datetime.today() - timedelta(days=int(1*365.25))
        end_date = datetime.today()
    elif start_date is None and end_date is None and no_of_years is not None:
        start_date = datetime.today() - timedelta(days=int(no_of_years*365.25))
        end_date = datetime.today()
    elif start_date is None and end_date is not None and no_of_years is not None:
        start_date = end_date - timedelta(days=int(no_of_years*365.25))
    elif start_date is not None and end_date is None and no_of_years is None:
        end_date = datetime.today()
    elif start_date is not None and end_date is None and no_of_years is not None:
        end_date = start_date + timedelta(days=int(no_of_years*365.25))
    elif start_date is not None and end_date is not None and no_of_years is not None:
        raise ValueError("Invalid date parameters.")
    elif start_date is None and end_date is not None and no_of_years is None:
        start_date = end_date - timedelta(days=int(no_of_years*365.25))
    
    if start_date is not None and end_date is not None and start_date >= end_date:
        raise ValueError("Start date should be before end date.")

    for t in tickers:
        adj_close[t] = yf.download(t, start_date, end_date, interval=interval)['Adj Close']
    
    print(f'\nThe following analysis is based on daily adjusted closing price data from {(start_date).strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}:')
    
    return adj_close, tickers
```

**Calculate returns, covariance and correlation between each stock:**
```python
def return_stats(adj_close, tickers):
    """
    Calculate various statistics related to returns and volatility of the assets.

    Args:
        adj_close (pandas.DataFrame): DataFrame containing adjusted closing prices of the assets.
        tickers (list): List of ticker symbols for the assets.

    Returns:
        pandas.DataFrame: DataFrame containing log returns of the assets.
        pandas.DataFrame: DataFrame containing annualized returns of the assets.
        pandas.DataFrame: DataFrame containing covariance matrix of the assets' returns.
        pandas.DataFrame: DataFrame containing correlation matrix of the assets' returns.
    """
    returns = np.log(1 + adj_close.pct_change()).dropna()
    returns_ann = returns * 250
    returns_cov = returns.cov() * 250
    returns_corr = returns.corr()
    
    print(f'\nAnnualized Total Returns (Daily):')
    for ticker in tickers:
        print(f'  {ticker}: {returns_ann[ticker].mean()*100:.2f}%')
    print()

    print(f'Annual Volatility:')
    for ticker in tickers:
        print(f'  {ticker}: {(returns[ticker].std() * 250 ** 0.5)*100:.2f}%')
    print()

    print(f'Correlation matrix: \n{np.round(returns_corr,3)}\n')

    return returns, returns_ann, returns_cov, returns_corr
```

**Retrieve risk free rate from Federal Reserve Economic Data (FRED) provided by the Federal Reserve Bank of St. Louis. using Fred API:**
```python
from fredapi import Fred

def retrieve_risk_free_rate(api_key=None):
    """
    Retrieve the risk-free rate.

    Args:
        api_key (str, optional): API key for data source. Defaults to None.

    Returns:
        float: Risk-free rate.
    """
    api_key = api_key or '5dbe996305fb72e106205e372aa30fb8'
    fred = Fred(api_key=api_key)
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    return risk_free_rate
```
"GS10" refers to the 10-year Treasury Constant Maturity Rate. This is the interest rate at which the U.S. government bonds with a maturity of approximately 10 years are issued (assumed to be the risk free rate)

**Calculate Portfolio return, volatility and sharpe ratio:**
```python
def portfolio_stats(weights, returns_ann, returns_cov, risk_free_rate):
    """
    Calculates portfolio statistics including return, volatility, and Sharpe ratio.

    Args:
        weights (numpy.ndarray): Array of portfolio weights.
        returns_ann (pandas.DataFrame): DataFrame of annualized returns for each asset.
        returns_cov (pandas.DataFrame): DataFrame of covariance matrix of returns.
        risk_free_rate (float): Risk-free rate of return.

    Returns:
        float: Portfolio return.
        float: Portfolio volatility.
        float: Sharpe ratio.
    """
    portfolio_return = np.dot(weights, returns_ann.mean())
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio
```

## Monte Carlo Efficient Frontier Simulation
To plot the Efficient frontier, multiple iterations of the portfolio with varying weights are required. The output needs to be in a dataframe - risk and volatility being the two columns. 

>The **np.random.random()** is used to obtain random floats counting equal to the count of tickers. Each float is then divided by the numpy sum of the floats to get the weights, such that the sum of weights = 1. The logic is $\frac{a}{a+b} + \frac{b}{a+b} = 1$.

**A DataFrame with simulated weights, their corresponding portfolio returns, volatilities and sharpe ratios is constructed in order to plot the efficient frontier of the portfolio:**
```python
def eff_frontier(tickers, returns_ann, returns_cov, risk_free_rate, no_of_iterations=1000):
    """
    Simulate multiple portfolios with varying weights to plot the efficient frontier.

    Args:
        tickers (list): List of ticker symbols for the assets.
        returns_ann (pandas.DataFrame): DataFrame containing annualized returns of the assets.
        returns_cov (pandas.DataFrame): DataFrame containing covariance matrix of the assets' returns.
        risk_free_rate (float): Risk-free rate.
        no_of_iterations (int, optional): Number of iterations for simulation. Defaults to 1000.

    Returns:
        numpy.ndarray: Array of weights for the simulated portfolios.
        numpy.ndarray: Array of portfolio returns.
        numpy.ndarray: Array of portfolio volatilities.
        numpy.ndarray: Array of Sharpe ratios for the simulated portfolios.
        pandas.DataFrame: DataFrame containing simulated portfolios.
    """
    pfolio_return = []
    pfolio_volatility = []
    weights_list = []

    for i in range (no_of_iterations):
        no_of_tickers = len(tickers)
        random_floats = np.random.random(no_of_tickers)
        weights = random_floats/np.sum(random_floats)
        
        weights_list.append(weights)
        pfolio_return.append(np.dot(returns_ann.mean(), weights))
        pfolio_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights))))

    weights = np.array(weights_list)
    pfolio_return = np.array(pfolio_return)*100
    pfolio_volatility = np.array(pfolio_volatility)*100
    sharpe_ratios = (pfolio_return - risk_free_rate) / pfolio_volatility

    eff_front = pd.DataFrame({
        'Portfolio Return':pfolio_return, 
        'Standard Deviaton':pfolio_volatility, 
        'Weights':weights_list, 
        'Sharpe Ratios':sharpe_ratios
    })
    
    return weights, pfolio_return, pfolio_volatility, sharpe_ratios, eff_front
```

From the simulated data, portfolios with the highest return, highest sharpe ratio and lowest volatility can be identified along with with their corresponding weights. 

**Plot:**

```python
def eff_frontier_plot(tickers, weights, pfolio_return, pfolio_volatility, sharpe_ratios, eff_front, risk_free_rate):
    """
    Plots the Efficient Frontier of the portfolio and highlights key portfolios.

    Args:
        tickers (list): List of ticker symbols.
        weights (numpy.ndarray): Array of portfolio weights.
        pfolio_return (numpy.ndarray): Array of portfolio returns.
        pfolio_volatility (numpy.ndarray): Array of portfolio volatilities.
        sharpe_ratios (numpy.ndarray): Array of Sharpe ratios.
        eff_front (pandas.DataFrame): DataFrame containing portfolio statistics for efficient frontier portfolios.
        risk_free_rate (float): Risk-free rate of return.
    """    
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe = sharpe_ratios[max_sharpe_idx]
    return_at_max_sharpe = pfolio_return[max_sharpe_idx]
    volatility_at_max_sharpe = pfolio_volatility[max_sharpe_idx]
    weights_at_max_sharpe = weights[max_sharpe_idx]
    
    min_volatility_idx = np.argmin(pfolio_volatility)
    min_volatility = pfolio_volatility[min_volatility_idx]
    return_at_min_volatility = pfolio_return[min_volatility_idx]
    weights_at_min_volatility = weights[min_volatility_idx]
    sharpe_ratio_at_min_volatility = sharpe_ratios[min_volatility_idx]
    
    max_return_idx = np.argmax(pfolio_return)
    max_return = pfolio_return[max_return_idx]
    vol_at_max_return = pfolio_volatility[max_return_idx]
    weights_at_max_return = weights[max_return_idx]
    sharpe_ratio_at_max_return = sharpe_ratios[max_return_idx]
    
    eff_front.plot(x='Standard Deviaton' , y='Portfolio Return', kind='scatter', figsize=(10,6))
    plt.scatter(volatility_at_max_sharpe, return_at_max_sharpe, marker='*', s=200, color='r', label='Max Sharpe Ratio')
    plt.scatter(min_volatility, return_at_min_volatility, marker='*', s=200, color='g', label='Minimum Volatility')
    
    plt.title(f'Efficient Frontier of the porfolio')
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.xlabel('Standard Deviation (%)')
    plt.ylabel('Portfolio Return (%)')
    plt.legend()
    plt.show()
    
    print(f'Efficient Frontier Portfolios:\n\n(a) Minimum volatility portfolio: \n  1. Weights:')
    for ticker, weight in zip(tickers, weights_at_min_volatility):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Portfolio Return: {return_at_min_volatility:.2f}%\n  3. Portfolio Volatility: {min_volatility:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_min_volatility:.4f}\n')
    
    print(f'(b) Maximum return portfolio: \n  1. Weights:')
    for ticker, weight in zip(tickers, weights_at_max_return):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Expected Annual Return: {max_return:.2f}%\n  3. Expected Volatility: {vol_at_max_return:.2f}%\n  4. Sharpe Ratio:{sharpe_ratio_at_max_return:.4f}\n')
    
    print(f'(c) Maximum Sharpe Ratio portfolio: \n  1. Weights:')
    for ticker, weight in zip(tickers, weights_at_max_sharpe):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Portfolio Return: {return_at_max_sharpe:.2f}%\n  3. Portfolio Volatility: {volatility_at_max_sharpe:.2f}%\n  4. Sharpe Ratio:{max_sharpe:.4f}\n')
    
    return
```

## Optimization with Scipy

Using the **minimize()** function from **Scipy optimize** module, the portfolio optimized for the highest sharpe ratio (under some constraints) can be obtained. The Sequential Least Squares Programming (SLSQP) method is used here. <br><br>

**Optimize for minimum volatility and a target return of 20.0%**


```python
from scipy.optimize import minimize

def opt_portfolio_min_volatility(tickers, returns_ann, returns_cov, risk_free_rate, bounds_per_ticker, min_return):
    """
    Optimizes portfolio for minimum volatility and a target return using Scipy minimize function.

    Args:
        tickers (list): List of ticker symbols.
        returns_ann (pandas.DataFrame): DataFrame of annualized returns for each asset.
        returns_cov (pandas.DataFrame): DataFrame of covariance matrix of returns.
        risk_free_rate (float): Risk-free rate of return.
        bounds_per_ticker (list): List of tuples containing bounds for portfolio weights.
        min_return (float): Target minimum return.

    Returns:
        numpy.ndarray: Array of optimized portfolio weights.
    """
    
    def min_constraints(weights):
        return np.sum(weights) - 1
    
    def min_return_constraint(weights):
        portfolio_return, _, _ = portfolio_stats(weights, returns_ann, returns_cov, risk_free_rate)
        return portfolio_return - min_return
    
    constraints = [{'type': 'eq', 'fun': min_constraints}, {'type': 'ineq', 'fun': min_return_constraint}]
    bounds = bounds_per_ticker * len(tickers)
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    
    def objective_function(weights):
        _, portfolio_volatility, _ = portfolio_stats(weights, returns_ann, returns_cov, risk_free_rate)
        return portfolio_volatility  
    
    optimized_results = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)
    return optimized_results.x
```

**Plot the optimized porfolio weights in a pie chart:**
```python
def opt_portfolio_plot(tickers, optimal_weights, returns_ann, returns_cov, risk_free_rate, min_return):
    """
    Plots the optimized portfolio weights and displays portfolio statistics.

    Args:
        tickers (list): List of ticker symbols.
        optimal_weights (numpy.ndarray): Array of optimized portfolio weights.
        returns_ann (pandas.DataFrame): DataFrame of annualized returns for each asset.
        returns_cov (pandas.DataFrame): DataFrame of covariance matrix of returns.
        risk_free_rate (float): Risk-free rate of return.
        min_return (float): Target minimum return for the optimized portfolio.

    Returns:
        None
    """
    optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio = portfolio_stats(optimal_weights, returns_ann, returns_cov, risk_free_rate)
    
    print(f'Optimized Portfilio:\n\n(a) The portfolio optimized for minimum volatility and a target return of {min_return*100}% (subject to constraints): \n  1. Weights:')
    for ticker, weight in zip(tickers, optimal_weights):
        print (f'      {ticker}: {weight:.3f}')
    print(f'  2. Expected Annual Return: {optimal_portfolio_return*100:.2f}%\n  3. Expected Volatility: {optimal_portfolio_volatility*100:.2f}%\n  4. Sharpe Ratio: {optimal_sharpe_ratio:.4f}\n')

    plt.figure(figsize=(2, 2))
    plt.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Optimized Portfolio Weights')
    plt.show()
    
    return
```

## Running the optimization
The program is run for a portfolio of Apple and Microsoft stocks based on their adjusted closing prices over the past 10 years.


```python
def main():
    """
    Main function to execute portfolio optimization. The iterable parameters are
        - tickers (list of str): List of ticker symbols representing the assets in the portfolio.
        - start_date (str or datetime): Start date for retrieving historical data. Format: 'YYYY-MM-DD'.
        - end_date (str or datetime): End date for retrieving historical data. Format: 'YYYY-MM-DD'.
        - no_of_years (int): Number of years of historical data to retrieve if start_date and end_date are not provided.
        - no_of_iterations (int): Number of iterations for simulating the efficient frontier.
        - bounds_per_ticker (list of tuples): List of tuples containing bounds for each asset's weight in the portfolio.
        - min_return (float): Target minimum return for the optimized portfolio.
    """
    tickers = ['AAPL', 'WMT']
    start_date = None
    end_date = None
    no_of_years = 10
    no_of_iterations = 1000
    bounds_per_ticker = [(0,1)]
    min_return = 0.20  
    
    adj_close_df, tickers = retrieve_data(tickers, start_date=start_date, end_date=end_date, no_of_years=no_of_years)
    returns, returns_ann, returns_cov, returns_corr = return_stats(adj_close_df, tickers)
    
    risk_free_rate = retrieve_risk_free_rate()
    
    weights, pfolio_return, pfolio_volatility, sharpe_ratios, eff_front = eff_frontier(tickers, returns_ann, returns_cov, risk_free_rate, no_of_iterations)
    eff_frontier_plot(tickers, weights, pfolio_return, pfolio_volatility, sharpe_ratios, eff_front, risk_free_rate)
      
    optimal_weights = opt_portfolio_min_volatility(tickers, returns_ann, returns_cov, risk_free_rate, bounds_per_ticker, min_return)
    opt_portfolio_plot(tickers, optimal_weights, returns_ann, returns_cov, risk_free_rate, min_return)
    
if __name__ == "__main__":
    main()

```
## Outputs
**The above code received the following outputs:**
```
The following analysis is based on daily adjusted closing price data from 2014-03-26 to 2024-03-25:

Annualized Total Returns (Daily):
  AAPL: 23.03%
  WMT: 10.77%

Annual Volatility:
  AAPL: 28.23%
  WMT: 20.73%

Correlation matrix: 
       AAPL    WMT
AAPL  1.000  0.322
WMT   0.322  1.000
```
![Efficient Frontier of the Portfolio](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/Eff%20Frontier.png)

```
Efficient Frontier Portfolios:

(a) Minimum volatility portfolio: 
  1. Weights:
      AAPL: 0.285
      WMT: 0.715
  2. Portfolio Return: 14.26%
  3. Portfolio Volatility: 19.01%
  4. Sharpe Ratio:0.7482

(b) Maximum return portfolio: 
  1. Weights:
      AAPL: 1.000
      WMT: 0.000
  2. Expected Annual Return: 23.03%
  3. Expected Volatility: 28.22%
  4. Sharpe Ratio:0.8145

(c) Maximum Sharpe Ratio portfolio: 
  1. Weights:
      AAPL: 0.652
      WMT: 0.348
  2. Portfolio Return: 18.76%
  3. Portfolio Volatility: 21.82%
  4. Sharpe Ratio:0.8579

Optimized Portfilio:

(a) The portfolio optimized for minimum volatility and a target return of 20.0% (subject to constraints): 
  1. Weights:
      AAPL: 0.753
      WMT: 0.247
  2. Expected Annual Return: 20.00%
  3. Expected Volatility: 23.41%
  4. Sharpe Ratio: 0.6746
```

![Optimized Portfolio Weights](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Python/Opt%20Port%20Weights.png)


## Analysis

The following conclusions can be drawn based on the above analysis of Apple (AAPL) and Walmart (WMT) stocks:

- **Return and Volatility:** Apple has a significantly higher annualized return ($23.03$%) compared to Walmart ($10.77$%). However, Apple also exhibits higher volatility ($28.23$%) compared to Walmart ($20.73$%), implying that it carries more risk. <br>
  
- **Correlation:** The correlation coefficient between Apple and Walmart stocks is $0.322$, indicating a positive but relatively low correlation between the two.<br>
    
- **Diversification Benefits:** Combining Apple and Walmart stocks in a portfolio offers diversification benefits due to their low correlation. The minimum volatility portfolio demonstrates this, with a significant allocation to WMT to balance the higher volatility of AAPL. <br>
  
- **Risk-Return Tradeoff:** The maximum return portfolio provides the highest return but at the cost of higher volatility. On the other hand, the minimum volatility portfolio sacrifices some return for lower volatility.

- **Risk adjusted return:** Sharpe ratio measures the risk adjusted relative returns of the portfolio. The maximum Sharpe ratio portfolio allocates a higher proportion to AAPL due to its higher return potential, while still benefiting from diversification with an allocation to WMT.

- **Optimization:** The optimized portfolio achieves the target return of $20.0$% with a balanced allocation between AAPL and WMT. This allocation aims to strike a balance between return and volatility while considering the correlation between the two stocks.



## Conclusion
The analysis underscores the fundamental risk-return tradeoff in investing. Portfolios with higher expected returns typically comes with increased volatility. 
