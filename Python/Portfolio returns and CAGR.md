# Portfolio Returns and CAGR Analysis with yfinance

## Table of Contents
1. [Introduction](#introduction)
2. [Individual Stock Analysis](#individual-stock-analysis)
    - [Simple Rate of Return](#simple-rate-of-return)
    - [Log Return](#log-return)
    - [Cumulative Return and CAGR](#cumulative-return-and-cagr)
3. [Portfolio Analysis](#portfolio)
    - [Comparing Relative Price Movements](#comparing-relative-price-movements:)
    - [Weighted Average Rate of Return](#calculating-weighted-average-rate-of-return:)
    - [Cumulative Return and CAGR](#cumulative-return-and-cagr:)
4. [Additional Notes](#some-additional-notes)
    - [Normalisation to 100](#normalisation-to-100)
    - [Alternative to using .shift()](#alternative-to-using-shift)

## Introduction

This project, originally created on Jupyter Notebook, delves into the comprehensive evaluation of a stock portfolio comprising Microsoft, Apple, Google, and Meta. 
  
  Leveraging the **yfinance** library in Python, the analysis extracts financial data from Yahoo Finance, aiming to showcase the calculation of essential metrics — **simple and log returns, cumulative returns, and Compound Annual Growth Rate (CAGR)** — with a focus on individual stocks as well as the portfolio as a whole.

The yfinance library provides a straightforward approach to downloading financial data, presented as a Pandas DataFrame.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
```

## Individual Stock Analysis 

```python
startDate = datetime.datetime(2023,3,2)
endDate = datetime.datetime(2024,3,2)

MSFT = yf.Ticker('MSFT')
MSFT_data = MSFT.history(start=startDate, end=endDate)
```

- The **'Ticker'** class is used to represent a specific financial instrument or asset, such as a stock. 
- The **history()** Method of the Ticker class is used to retrieve historical market data for the specified asset. 


```python
MSFT_data.head()
```

In general, it is preferable to use **simple returns** when calculating the returns of multiple securities in the same period. **Log returns** are a better choice when calculating the return of a single security over multiple time periods.

### 1. Simple Rate of Return 

Simple Return is given by the formula: $$\( \frac{P_1 - P_0}{P_0} = \frac{P_1}{P_0} - 1 \)$$

The **.pct_change()** method calculates the fractional change between the current and a prior element. Here it is used to calculate simple returns. 

```python
MSFT_data1 = MSFT_data.copy()

# Adding the column 'Daily Simple Returns' to the original dataframe
MSFT_data1['Daily Simple Returns'] = MSFT_data1['Close'].pct_change()
MSFT_data1 = MSFT_data1.dropna()

print (MSFT_data1['Daily Simple Returns'])
```


```python
MSFT_data1['Daily Simple Returns'].plot(figsize=(15,5))
plt.title("Daily Simple Returns of MSFT")
plt.show
```

By default, using **.plot()** on a pandas DataFrame or Series creates a 'line' plot. To obtain a different type of plot, specify it by using the 'kind' parameter. 


```python
avg_returns_daily = MSFT_data1['Daily Simple Returns'].mean()

# Assuming 250 trading days in a year
avg_returns_annualized = avg_returns_daily * 250
avg_returns_annualized
```


```python
print(f'The average daily simple returns of MSFT in the given period is {round(avg_returns_daily,4)*100}%')
print(f'The annualized average daily simple returns of MSFT in the given period is {round(avg_returns_annualized,5)*100}%')
```

#### Putting it all into a function:


```python
def annualized_simple_return(ticker, StartDay, StartMonth, StartYear, EndDay, EndMonth, EndYear, trading_days_in_year=250):
    """
    Calculate and visualize the daily and annualized simple returns of a stock.

     Parameters:
    - ticker (str): Stock symbol.
    - StartDay, StartMonth, StartYear (int): Start date for data retrieval.
    - EndDay, EndMonth, EndYear (int): End date for data retrieval.
    - trading_days_in_year (int): Number of trading days in a year (default is 250).

    Returns:
    None
    """
    # Convert input parameters to datetime objects
    start = datetime.datetime(StartYear, StartMonth, StartDay)
    end = datetime.datetime(EndYear, EndMonth, EndDay)

    # Retrieve historical data for the specified stock and period
    asset_data = yf.Ticker(ticker).history(start=start, end=end)

    # Calculate daily simple returns
    asset_data['Daily Simple Return'] = asset_data['Close'].pct_change()
    asset_data = asset_data.dropna()

    # Calculate average daily and annualized simple returns
    average_daily_return = asset_data['Daily Simple Return'].mean()
    average_annualized_return = average_daily_return * 250

    # Visualize the daily simple returns
    asset_data['Daily Simple Return'].plot(figsize=(15, 5))
    plt.title(f'{ticker} Daily Simple Returns from {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}')
    plt.show()

    # Print results
    print(f'The average daily simple return of {ticker} during the given period is {average_daily_return:.2%}')
    print(f'The annualized average daily simple return of {ticker} during the given period is {average_annualized_return:.2%}')

# Example usage
annualized_simple_return('MSFT', 2, 3, 2023, 2, 3, 2024)

```

### 2. Log Return 

Log return is given by the formula $$\( \ln \frac{P_t}{P_{t-1}} \)$$
However, since percentage changes can be close to zero or negative, the adjusted closing prices are added by 1 in order to avoid errors. 


```python
MSFT_data2 =  MSFT_data.copy()

#the log function is from the numpy package
MSFT_data2['Daily log returns'] = np.log(1 + MSFT_data2['Close'].pct_change())
MSFT_data2 = MSFT_data2.dropna()

print (MSFT_data2['Daily log returns'])
```


```python
MSFT_data2['Daily log returns'].plot(figsize=(15,5))
plt.title("Daily Log Returns of MSFT")
plt.show
```


```python
average_returns_daily = MSFT_data2['Daily log returns'].mean()

# Assuming 250 trading days in a year
log_returns_annualized = average_returns_daily * 250
log_returns_annualized
```


```python
print (str(round(log_returns_annualized,5)*100) + '%')
```

#### Putting it all into a function:


```python
def annualized_log_returns(ticker, StartDay, StartMonth, StartYear, EndDay, EndMonth, EndYear, trading_days_in_year=250):
    """
    Calculate and display the average daily log returns and annualized average daily log returns for a given stock.

    Parameters:
    - ticker (str): Stock symbol.
    - StartDay, StartMonth, StartYear (int): Start date for data retrieval.
    - EndDay, EndMonth, EndYear (int): End date for data retrieval.
    - trading_days_in_year (int): Number of trading days in a year (default is 250).

    Returns:
    None
    """
    # Data retrieval
    start = datetime.datetime(StartYear, StartMonth, StartDay)
    end = datetime.datetime(EndYear, EndMonth, EndDay)
    asset_data = yf.Ticker(ticker).history(start=start, end=end)

    # Calculate daily log returns
    asset_data['Daily Log Returns'] = np.log(1 + asset_data['Close'].pct_change())
    asset_data = asset_data.dropna()

    # Calculate average daily log returns
    average_return = asset_data['Daily Log Returns'].mean()

    # Calculate annualized average daily log returns
    average_return_annualized = average_return * trading_days_in_year

    # Plot daily log returns
    asset_data['Daily Log Returns'].plot(figsize=(15, 5))
    plt.title(f'{ticker} Daily Log Returns from {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}')
    plt.show()

    # Display results
    print(f'The average daily log returns of {ticker} during the given period is {average_return:.2%}')
    print(f'The annualized average daily log returns of {ticker} during the given period is {average_return_annualized:.2%}')

# Example usage with 252 trading days in a year
annualized_log_returns('MSFT', 2, 3, 2023, 2, 3, 2024, trading_days_in_year=252)
```

### 3. Cumulative Return and CAGR

The <u> **cumulative return** </u> is the total percentage change in the price from the start date to the end date. It is a measure of the overall investment performance over a specific period. The **cumprod()** function is used for this purpose. 
   - The pct_change() method computes the percentage change between the each day's adjusting closing prices. In order to calculate cumulative return, 1 is added to each of these percentages, converting them into multipliers.
   - After calculating the cumulative product, subtracting 1 adjusts for the initial value of 1 that was added. This gives the total relative return over the period.

**Compound annual growth rate**, or CAGR, is the mean annual growth rate of an investment over a specified period of time longer than one year.

$$ \ \left( \frac{\text{Ending Value}}{\text{Beginning Value}} \right) ^ {\frac{1}{\text{n periods}}} - 1 \  $$  
   - cumprod.iloc[-1] corresponds to the cumulative product at the end of the investment period, which represents the total growth of the investment from the start to the end of the period.
   - It is then raised to the power of the reciprocal of the number of years. This step calculates the geometric mean return, which is the nth root of the product of n numbers. In this case, it represents the average growth rate per year.
   - The subtraction of 1 is a convention in the CAGR formula to express the result as a percentage. 


```python
def calculate_cumulative_return_and_cagr(ticker, startday, startmonth, startyear, endday, endmonth, endyear):
    """
    Calculate the cumulative simple return and Compound Annual Growth Rate (CAGR) for a given stock.

    Parameters:
    - ticker (str): Ticker symbol of the stock.
    - startday, startmonth, startyear (int): Start date of the analysis period.
    - endday, endmonth, endyear (int): End date of the analysis period.
    """
    # Convert input to datetime objects
    startdate = datetime.datetime(startyear, startmonth, startday)
    enddate = datetime.datetime(endyear, endmonth, endday)
    
    # Fetch historical stock data
    data = yf.Ticker(ticker).history(start=startdate, end=enddate)
    
    # Calculate daily simple returns
    simple_returns = data['Close'].pct_change() + 1
    simple_returns = simple_returns.dropna()
    
    # Calculate cumulative product and obtain cumulative return
    cumprod = simple_returns.cumprod()
    cumulative_return = cumprod.iloc[-1] - 1
    
    # Calculate the number of years for CAGR calculation
    # Extract the number of days between enddate and startdate. 
    # The resulting value is divided by 365.25 to convert it into the equivalent number of years, accounting for leap years. 
    years = (enddate - startdate).days / 365.25
    
    # Calculate CAGR
    cagr = (cumprod.iloc[-1])**(1/years) - 1
    
    # Plot adjusted closing price
    data['Close'].plot(figsize=(15,5))
    plt.title(f'Adjusted closing price from {startdate.strftime("%Y-%m-%d")} to {enddate.strftime("%Y-%m-%d")}')
    plt.show()
    
    # Print results
    print(f'The cumulative simple return of {ticker} during the period is {round(cumulative_return * 100, 2)}%')
    print(f'The CAGR of {ticker} during the period is {round(cagr * 100, 2)}%')
    return

# Example usage
calculate_cumulative_return_and_cagr('MSFT', 2, 3, 2019, 2, 3, 2024)

```

## <u> Portfolio

The function **yf.download()** is convenient to fetch stock data of multiple stocks in the portfolio. 

### Comparing relative price movements:


```python
tickers_list = ['MSFT', 'AAPL', 'GOOG', 'META']

startDate = datetime.datetime(2015,1,1) 
endDate = datetime.datetime(2023,12,31)

# Import data into empty dataframe in the order of Ticker list. 
stock_data = pd.DataFrame()
for t in tickers_list:
    stock_data[t] = yf.download(t, startDate, endDate)['Close']

# Normalisation to base 100 before plotting
((stock_data/stock_data.iloc[1])*100).plot(figsize=(15,7))

plt.legend()
plt.title("Comparing Relative Stock Price movement", fontsize=16)
plt.ylabel('$', fontsize=14)
plt.xlabel('Years', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5) 
plt.xlim(left=stock_data.index[0])
plt.show
```

### Calculating weighted average rate of return:

Simple return of each stock is multiplied by their respective weights in the portfolio. The weights can represent the proportion of the total portfolio value that each stock contributes.
The **NumPy dot product** performs the weighted sum of the individual stock returns, giving the portfolio return. 


```python
weights = np.array([0.25,0.25,0.25,0.25])

daily_returns = stock_data.pct_change()
daily_returns = daily_returns.dropna()

average_daily_returns = daily_returns.mean()
annualized_average = average_daily_returns*250

portfolio_return = np.dot(annualized_average, weights)
portfolio_return
```

### Calculating cumulative return and CAGR:

- The weighted cumulative growth is calculated by taking the dot product of the final cumulative product values and the specified weights.


```python
cumproduct = (daily_returns + 1).cumprod() - 1
no_of_years = (endDate -  startDate).days/365.25

cumulative_return = np.dot(cumproduct.iloc[-1], weights)
CAGR_portfolio = cumulative_return**(1/no_of_years) - 1

cumulative_return, CAGR_portfolio
```

### Putting it all into a function:


```python
def portfolio_return_cagr(tickers, weights, start_date, end_date, trading_days_in_year=250):
    """
    Calculate the portfolio's weighted average return, cumulative return and CAGR, and visualize relative stock price movement.

    Parameters:
    - tickers (list): List of stock tickers in the portfolio.
    - weights (np.array): Array of weights corresponding to each stock in the portfolio.
    - start_date (datetime): Start date for the analysis period.
    - end_date (datetime): End date for the analysis period.
    - trading_days_in_year (int, optional): Number of trading days in a year. Default is 250.

    Returns:
    - None
    """

    # Download stock data for each ticker
    data = pd.DataFrame()
    for t in tickers:
        data[t] = yf.download(t, start_date, end_date)['Close']

    # Plotting relative stock price movement
    stock_price_normalized = ((data / data.iloc[0]) * 100)
    stock_price_normalized.plot(figsize=(15, 7))
    plt.legend()
    plt.title("Comparing Relative Stock Price Movement", fontsize=16)
    plt.ylabel('$', fontsize=14)
    plt.xlabel('Years', fontsize=14)
    plt.xlim(left=data.index[0])
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()

    # Calculate daily returns
    daily_returns = data.pct_change()
    daily_returns = daily_returns.dropna()

    # Calculate average daily returns and annualize
    average_daily_returns = daily_returns.mean()
    annualized_average = average_daily_returns * trading_days_in_year

    # Calculate the portfolio return using weights
    portfolio_return = np.dot(annualized_average, weights)
        
    # Calculate cumulative return and CAGR
    cumproduct = (daily_returns + 1).cumprod() - 1
    no_of_years = (end_date -  start_date).days/365.25

    cumulative_return = np.dot(cumproduct.iloc[-1], weights)
    CAGR_portfolio = cumulative_return**(1/no_of_years) - 1

    # Print outputs
    print(f'The weighted average return of the portfolio during the given period is {portfolio_return:.2%}')
    print(f'The cumulative return of the portfolio during the given period is {cumulative_return:.2%}')
    print(f'The compound annual growth rate of the portfolio during the given period is {CAGR_portfolio:.2%}')
        
    return 

# Example usage
tickers_list = ['MSFT', 'AAPL', 'GOOG', 'META']
weights = np.array([0.25, 0.25, 0.25, 0.25])
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2023, 12, 31)
portfolio_return_cagr(tickers_list, weights, start_date, end_date)

```

.

.

.

.

# Some additional notes

### 1. Normalisation to 100

A trick used so that all lines start from the same spot 100. The following equation on its first iteration divides value from row one by itself, giving 1, which is then multiplied by 100. In the line chart, all the lines will initiate from the same point. 

$$\( \frac{P_t}{P_0} \times 100 \) $$

Normalizing the stock prices to a base of 100 is a common practice when you want to compare the relative performance of multiple stocks over time, especially when their absolute values differ significantly. However, if your goal is to directly observe the actual price movements, normalizing can indeed make it harder to interpret the absolute price levels. 

$$ \( \text{((data/data.iloc[0])*100)} \) $$ 

> iloc is used to get the row using zero based indexing. Here we get the values of the first row 

The use of the cumulative product (**cumprod**) effectively avoids the need to normalize the returns to 100 by using a multiplier. When you multiply consecutive daily returns, you get a cumulative product that represents the total growth or decline over time.

If you were to normalize the returns to 100, you would start with an initial value of 100 and then apply the daily returns to calculate the new values. However, using the cumulative product achieves a similar outcome by directly representing the growth factor over time.

In the context of plotting cumulative returns, using the cumulative product is a common and convenient approach, as it directly shows how an investment would have grown or declined over the specified time period. It simplifies the process and provides a clear representation of the overall performance of the investments.

### 2. Alternative to using **.shift()**:  


Pandas.DataFrame.pct_change() computes simple returns directly.

s_rets_1 = (data / data.shift(1)) - 1

s_rets_2 = data.pct_change()

A few arguments can be used in the percentage change method. The most important one is **'period'** as it specifies the difference between prices in the nominator. By default, it equals one. Let's assume we would like to calculate simple returns with the following formula: 

$$ \( \frac{P_t - P_{t-2}}{P_{t-2}} \) $$

Then, we should specify 'periods = 2' in parentheses: 

#### Using .pct_change() to calculate cumulative rate of return of a porfolio:

The **.pct_change()** calculates the percentage change of the values in the DataFrame data.

**data.pct_change() + 1** adds 1 to the daily percentage change. This step is performed to convert the percentage change to a multiplier. For example, if the daily percentage change is 2%, adding 1 results in a multiplier of 1.02.

**.cumprod()** calculates the cumulative product of the above result. It essentially computes the cumulative returns over time (cumulative return multiplier day 2 = return multiplier day2 * return multiplier day1).

1 is subtracted from the cumulative returns in order to convert the cumulative returns back to a percentage form, making it more interpretable. Otherwise they would be in absolute terms. 

#### Using .pct_change() to calculate Log Returns

log_rets_1 = np.log(data / data.shift(1))

log_rets_2 = np.log(data.pct_change() + 1)

Mathematically, it will look like this:

$$
ln(\frac{P_t}{P_{t-1}} ) = ln( \frac{P_t - P_{t-1}}{P_{t-1}} + \frac{P_{t-1}}{P_{t-1}}) = ln(\ simple.returns + 1)
.$$

Mathematically, the logarithm of a non-positive or zero value is undefined. The logarithm function is not defined for values less than or equal to zero. Logarithms are only defined for positive real numbers.

In the context of financial data, taking the logarithm of a percentage change is a common transformation. However, if the percentage change is very close to zero or negative (for example, when the closing price remains the same or decreases), adding 1 before taking the logarithm ensures that the argument of the logarithm is always positive. The transformation becomes log(1 + x), where x is the percentage change. This prevents mathematical issues and makes the transformation suitable for a wider range of financial data.

While this transformation alters the raw percentage change values slightly, it often has minimal impact on the overall analysis. The purpose is to ensure the mathematical validity of the logarithmic transformation and maintain stability, especially when dealing with financial time series data.
