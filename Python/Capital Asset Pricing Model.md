# **Capital Asset Pricing Model (CAPM)**

## Table of Contents
1. [Introduction](#introduction)
   - [Expected Return](#expected-return-r_i)
   - [Beta](#beta-beta)
2. [CAPM with Python](#capm-with-python)
   - [1.Retrieve stock data and calculate log returns](#1-retrieve-stock-data-and-calculate-log-returns)
   - [2.Calculate Beta](#2-calculate-beta-beta_im)
   - [3.Estimate expected returns](#3-estimate-expected-returns-r_i)
3. [Sharpe Ratio](#sharpe-ratio)
   - [Calculate with Python](#calculate-with-python)
   - [Limitation](#limitation)
5. [Summary](#summary) 
6. [Additional Notes](#additional-notes)

## Introduction
This project provides a brief overview of the Capital Asset Pricing Model and its practical applications in financial analysis. Subsequently, it delves into the implementation of CAPM using Python, demonstrating how to calculate beta values and expected returns, and assess risk-adjusted returns using the Sharpe Ratio. 

Libraries required:
```python
import pandas as pd
import numpy as np
import yfinance as yf
```

### Expected return ($r_i$)
When an investor buys a share of a stock, they expect to earn a return that compensates them for the risk they're taking. <br><br>This compensation (return) is made up of a baseline risk-free return ($r_f$) and an additional return based on the risk of the stock, which is proportional to the overall market risk (equity risk premium), adjusted by the stock's sensitivity to market movements ($\beta_{im}$). <br> 

Thus the **expected return ($r_i$)** of a particular stock  is given by:

$$r_i = r_f + \beta_{im}(r_m-r_f)$$ 

where
>$r_f$ is the risk free rate <br>
>$\beta_{im}$ is the beta between the stock and the market <br>
>$r_m$ is the expected market return <br>
>$r_m - r_f$ is the equity risk premium

- **Risk free rate** is the return an investor would get from a completely safe, risk-free investment. In reality, there is always some level of risk, but for the sake of this formula, it is assumed there is a risk-free asset with no risk.

- **Expected market return** is the expected return of a well-diversified portfolio that represents the overall market. It's a measure of how the entire market is expected to perform. 

- **Equity risk premium** is the extra return that investors demand for holding a risky asset (like stocks) instead of a risk-free asset. It's essentially the additional compensation for taking on the risk associated with the stock market.



### **Beta ($\beta$)**

$\beta$ measures the sensitivity of a stock's returns to changes in the market.

$$\beta_x = \frac{{Cov}_{x,m}}{Var_m} $$ 
where 
> $x$ represents the asset <br>
> $m$ represents the market 

- A beta of 1 means the stock tends to move with the market. 
- A beta less than 1 indicates the stock is less volatile than the market.
- A beta greater than 1 indicates higher volatility.

## CAPM with Python 
### 1. Retrieve stock data and calculate log returns: 

```python
def periodic_stock_returns(tickers, period, interval):
    """
    Download historical stock prices and calculate periodic returns.

    Parameters:
    -----------
    - tickers (list): List of stock tickers
    - period (str): Historical data period (e.g. "5y" for 5 years)
    - interval (str): Interval for data (e.g. "1mo" for 1 month)

    Returns:
    --------
    - data (DataFrame): DataFrame containing historical stock prices
    - returns (DataFrame): DataFrame containing periodic returns
    """    
    # Download historical stock prices using yfinance
    data = yf.download(tickers, period=period, interval=interval)['Adj Close']
    
    # Calculate periodic returns 
    returns = np.log(1 + data.pct_change()).dropna()
    
    return data, returns

# Example Usage
tickers = ['WMT','KO','LMT','PFE']
periodic_stock_returns(tickers, period="5y", interval="1mo")
```
This example code returns the monthly adjusted closing prices and log returns of Walmart Inc, Coca-Cola Co, Lockheed Martin Corp and Pfizer Inc from the past 5 years.

### 2. Calculate Beta ($\beta_{im}$):
```python
def calculate_beta(tickers, market_index='^GSPC'):
    """
    Calculate beta values for given stocks with respect to a market index.

    Parameters:
    -----------
    - tickers (list): List of stock tickers
    - market_index (str): Ticker symbol of the market index (default: '^GSPC' for S&P 500)

    Returns:
    --------
    - beta_df (Series): Series containing beta values for each stock
    """  
    # Adding the market index to tickers list
    stocks = tickers + [market_index]

    # Download historical stock prices and returns
    stock_data = pd.DataFrame()
    stock_returns = pd.DataFrame()
    for t in stocks:
        try: 
            stock_data[t], stock_returns[t] = periodic_stock_returns(t)
        except Exception as e:
            print(f'Error downloading data for {t}: {e}')            

    # Calculate covariance matrix and extract market variance. 
    cov_matrix = stock_returns.cov() * 12
    var_market = cov_matrix.loc[market_index, market_index]

    # Calculate beta values
    beta_df = cov_matrix.loc[tickers, market_index]/var_market
    beta_df.name = 'Beta'

    return beta_df

# Example Usage
calculate_beta(tickers)
```
Note that Beta calculated in this example is 5Y monthly. This can be changed in the `periodic_stock_returns` function. 

### 3. Estimate expected returns ($r_i$):

A 10 year US government bond can be considered risk free. Its yield as of 13 March 2024 is 4.19% [(Bloomberg.com)](https://www.bloomberg.com/markets/rates-bonds/government-bonds/us) <br>


```python
def capm_expected_return(tickers, risk_free_rate, risk_premium, market_index='^GSPC', print_info=False):
    """
    Calculate expected returns using Capital Asset Pricing Model (CAPM).

    Parameters:
    -----------
    - tickers (list): List of stock tickers
    - risk_free_rate (float): Risk-free rate of return
    - risk_premium (float): Expected risk premium
    - market_index (str): Ticker symbol of the market index (default: '^GSPC' for S&P 500)
    - print_info (bool): Whether to print expected returns for each stock (default: False)

    Returns:
    --------
    - CAPM_return (Series): Expected returns calculated using CAPM
    """
    # Calculate beta for each stock using the provided market index
    beta = calculate_beta(tickers, market_index=market_index)
    
    # Calculate CAPM return for each stock
    CAPM_return = risk_free_rate + beta * risk_premium
    
    # Print the expected return for each stock
    if print_info:
        print(f'The expected return of the given stocks as per Capital Asset Pricing Model:\n')
        for t in tickers:
            print(f'{t}: {CAPM_return.loc[t]*100:.2f}%')
    
    return CAPM_return

# Example Usage
capm_expected_return(tickers, 0.0417, 0.05)
```

# **Sharpe Ratio**

The Sharpe Ratio is a measure of risk-adjusted return, commonly used to evaluate the performance of an investment or a portfolio. 
$$\frac{r_i-r_f}{\sigma_i}$$
where 
> $r_i$ is the average rate of return on the investment $i$.   <br>
> $r_f$ is the risk free rate. <br>
> $\sigma_i$ is the standard deviation of the investment. <br>

$r_i-r_f$ is the **excess return**, representing the return earned above the risk-free rate, while $\sigma_i$ is the **standard deviation** of the investment's returns. It serves as a measure of risk. 

By dividing the excess return by the risk, the Sharpe Ratio **quantifies how much return an investor is receiving per unit of risk taken**. 
Investors and fund managers often use the Sharpe Ratio to compare the risk-adjusted performance of different investments or portfolios. 
- Higher Sharpe Ratios are generally preferred, as they suggest a better trade-off between risk and return.
- It indicates if a certain investment fund is performing satisfactory on a risk adjusted basis (at the expense of a riskier portfolio.)

## Calculate with Python:
```python
def calculate_sharpe_ratio(tickers, risk_free_rate, risk_premium, market_index='^GSPC'):
    """
    Calculate Sharpe ratio for given stocks.

    Parameters:
    -----------
    - tickers (list): List of stock tickers
    - risk_free_rate (float): Risk-free rate of return
    - risk_premium (float): Expected risk premium
    - market_index (str): Ticker symbol of the market index (default: '^GSPC' for S&P 500)

    Returns:
    --------
    - sharpe_ratio (Series): Sharpe ratio for each stock
    """
    # Calculate Capital Asset Pricing Model (CAPM) expected returns for given tickers
    capm_exp_ret = capm_expected_return(tickers, risk_free_rate, risk_premium, market_index=market_index)
    
    # Calculate periodic returns for the given tickers
    _, returns = periodic_stock_returns(tickers)
    
    # Calculate the average returns of the stocks
    average_returns = returns.mean()
    
    # Calculate the standard deviation of the average returns and annualize it
    std_dev = average_returns.std() * 12 ** 0.5
    
    # Calculate the Sharpe ratio using the CAPM expected return and standard deviation
    sharpe_ratio = (capm_exp_ret - risk_free_rate) / std_dev
    
    return sharpe_ratio

# Example Usage
calculate_sharpe_ratio(tickers, 0.0417, 0.05)
```
## Limitation
One limitation of the provided code lies in the assumption made during the annualization of the standard deviation in the `calculate_sharpe_ratio` function. The code multiplies the standard deviation by the square root of 12 to convert it to an annualized measure, assuming that the `interval` parameter in the `periodic_stock_returns` function is set to "1mo" (monthly data). However, if it is set to any other value, such as "1d" for daily data or "1y" for yearly data, this assumption would lead to inaccurate annualization of the standard deviation. In such cases, the code would fail to appropriately adjust for the different intervals, potentially resulting in incorrect Sharpe Ratio calculations and misleading risk-adjusted performance assessments. <br>
To rectify this limitation, the code should dynamically adjust the annualization factor based on the actual interval specified in the `periodic_stock_returns` function. This adjustment can be achieved by introducing a conditional statement that selects the appropriate annualization factor corresponding to the chosen interval, ensuring accurate conversion of standard deviation to an annualized measure regardless of the data interval used.

# Summary
This project presents a thorough exploration of the Capital Asset Pricing Model (CAPM) and the Sharpe Ratio, along with their practical implementation in Python. It combines theoretical explanations with hands-on coding examples.

# ____________________________________________________________________________
<br><br><br><br><br><br><br>
# **Additional Notes**

## **Alpha ($\alpha$)**

Alpha is a key metric used to evaluate the performance of an investment. It represents the excess return generated beyond what would be expected based on the inherent risks associated with the market. The standard CAPM setting assumes efficient financial market and therefore an $\alpha$ of zero. It is included in the augmented CAPM expected return formula:

$$r_i = \alpha + r_f + \beta_{im}(r_m-r_f)$$ 

- Given that $\beta$ multiplied by the equity risk premium gives us the compensation for risk that's been taken with the investment, $\alpha$ shows us how much return we get without bearing extra risk. 
- It provides insight into the baseline return, acting as a measure of intrinsic performance of an investement fund beyond market dynamics. Positive or negative values of $\alpha$ suggest either outperformance or underperformance relative to market expectations. 
- A practical implication is that a fund manager charging fees equivalent to 1% of the invested amount would need an $\alpha$ exceeding 1% to rationalize the associated costs.

Note that alpha is comparable only when the risk profile (beta) of the investments being compared is similar.

## Limitations of CAPM and Sharpe Ratio:

1. Assumptions of CAPM:
> CAPM relies on several assumptions, including perfect market conditions, rational investor behavior, and linear relationships between risk and return. In reality, these assumptions may not hold true, leading to potential inaccuracies in expected returns and risk assessments.  

2. Single-Factor Model:
> CAPM is a single-factor model that considers only systematic risk (beta) in determining expected returns. It ignores other relevant factors that may influence asset prices, such as company-specific events, macroeconomic indicators, and geopolitical risks. This limitation can result in mispricing of assets, especially during periods of market turbulence or structural changes.

3. Market Efficiency:
> CAPM assumes market efficiency, implying that all available information is reflected in asset prices. However, markets may not always be efficient, leading to mispricings and inefficiencies that CAPM fails to account for. Behavioral biases, information asymmetry, and market anomalies can challenge the validity of CAPM's predictions.

4. Homogeneous Investor Preferences:
> CAPM assumes that all investors have homogeneous risk preferences and investment horizons. In reality, investors exhibit diverse risk tolerances, investment objectives, and time horizons, leading to variations in asset valuations and expected returns. CAPM's homogeneity assumption oversimplifies investor behavior, potentially leading to inaccurate risk assessments.

5. Non-Normal Distributions:
> The Sharpe Ratio, like many risk-adjusted performance measures, assumes a normal distribution of returns. However, financial markets often exhibit non-normal distributions, including fat tails, skewness, and kurtosis. Non-normality can distort risk estimates and misrepresent the true risk-return profile of investments, undermining the reliability of the Sharpe Ratio.

6. Sensitivity to Inputs:
> Both CAPM and the Sharpe Ratio are sensitive to input parameters, such as the risk-free rate, market index selection, and estimation periods. Small changes in these inputs can lead to significant variations in model outputs, affecting investment decisions and performance evaluations.

7. Historical Data Limitations:
> CAPM and the Sharpe Ratio rely on historical data to estimate future returns and risk. However, past performance may not accurately predict future outcomes, especially during periods of structural shifts, regulatory changes, or technological innovations. Overreliance on historical data may result in suboptimal investment decisions and risk management strategies.
  
Addressing these limitations requires a nuanced understanding of financial markets, incorporating additional risk factors, utilizing alternative models, and employing robust risk management techniques. While CAPM and the Sharpe Ratio offer valuable insights into asset pricing and performance evaluation, investors should supplement these models with qualitative analysis and judgment to make informed investment decisions.

