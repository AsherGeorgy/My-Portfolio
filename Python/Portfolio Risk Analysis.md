```python
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
```

# **Risk Analysis**
<hr>

Variability is the best measure of risk. A volatile stock is much more likely to deviate from its historical returns and surprise investors negatively. <br>
  
**Standard deviation** helps quantify risk associated with the dispersion (volatitlity). 


## <u> Individual Stock Analysis

### 1. Comparing Risk & Return of stocks:

Three functions need to be defined for this purpose:
- `download_stock_data` to download adjusted closing prices.
- `calculate_returns` to calculate log returns.
- `calculate_risk_return` to calculate the stocks' volatility (risk) and return.

```python
def download_stock_data(tickers, start_date, end_date=None, plot_data=False):
    """
    Downloads adjusted close prices for selected stocks between two dates and plots them on a line chart normalized to 100. 

    Parameters:
    -----------
    - tickers (list): List of stock tickers to analyze.
    - start_date (datetime): Start date for historical data retrieval.
    - end_date (datetime): End date for historical data retrieval. Default is None, indicating the current date and time.
    - plot_data (bool): If true, plots the data; otherwise suppresses plotting (default is False).
    
    Returns:
    -------
    Pandas DataFrame 
    """
    # Use current date and time if end_date is not provided
    end_date = end_date or datetime.datetime.now()

    # Initialize DataFrame to store stock data
    stock_data = pd.DataFrame()
    
    try:
        # Download adjusted close prices for each stock
        for ticker in tickers:
            stock_data[ticker] = yf.download(ticker, start_date, end_date)['Adj Close']
    except Exception as e:
        raise RuntimeError(f"Error downloading stock data: {e}")
        
    if plot_data:
        ((stock_data/stock_data.iloc[0])*100).plot(figsize=(10, 6))
        plt.title('Relative Stock price movement')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid()
        plt.show()

    return stock_data

```


```python
def calculate_returns(tickers, start_date, end_date=None):
    """
    Calculates daily simple log returns of stock data.

    Parameters:
    -----------
    - tickers (list): List of stock tickers to analyze.
    - start_date (datetime): Start date for historical data retrieval.
    - end_date (datetime): End date for historical data retrieval. Default is None, indicating the current date and time.
    
    Returns:
    -------
    Pandas DataFrame 
    """
    # Use current date and time if end_date is not provided
    end_date = end_date or datetime.datetime.now()
    
    # Download adjusted close prices for each stock
    try:
        stock_data = download_stock_data(tickers, start_date, end_date=None)
    except Exception as e:
        raise RuntimeError(f"Error downloading stock data: {e}")
                                         
        # Calculate logarithmic returns
    simple_returns = np.log(stock_data.pct_change() + 1)
    simple_returns = simple_returns.dropna()
    
    return simple_returns
```


```python
def calculate_risk_return(tickers, start_date, end_date=None, trading_days_in_year=250, print_results=False):
    """
    Calculate stock performance in terms of risk (volatility) and return.

    Parameters:
    -----------
    - tickers (list): List of stock tickers to analyze.
    - start_date (datetime): Start date for historical data retrieval.
    - end_date (datetime): End date for historical data retrieval. Default is None, indicating the current date and time.
    - trading_days_in_year (int): Number of trading days in a year (default is 250).
    - print_results (bool): If True, prints variance information; otherwise, suppresses printing (default is False).
    
    Returns:
    -------
    Pandas DataFrame
    """
    # Use current date and time if end_date is not provided
    end_date = end_date or datetime.datetime.now()

    # Calculate simple returns of the stocks
    try:
        simple_returns = calculate_returns(tickers, start_date, end_date)
    except Exception as e:
        raise RuntimeError(f"Error calculating returns: {e}")

    # Calculating annualized average return and standard deviation
    annualized_average_return = simple_returns.mean() * trading_days_in_year
    annualized_std_dev = simple_returns.std() * (trading_days_in_year ** 0.5)

    # Print results
    if print_results:   
        for ticker in tickers:
            print(f'{ticker} - Annualized Average Return: {annualized_average_return[ticker]*100:.2f}% | Volatility: {annualized_std_dev[ticker]*100:.2f}%')

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Ticker': tickers,
        'Average Return': annualized_average_return, 
        'Volatility': annualized_std_dev
    })

    return results_df
```


```python
# Example Usage:
tickers = ['AAPL', 'GOOG']
start_date = datetime.datetime(2014, 1, 1)

calculate_risk_return(tickers, start_date, print_results=True)
```

### 2. Covariance and Correlation

There might exist a relationship between stock prices of different companies. Investigating what causes this relationship can help build optimal investment portfolios.  

- **Variance** is a statistical measure that quantifies the dispersion or spread of a set of values in a dataset.
- **Covariance** measures the degree to which two random variables change together. 
- **Correlation** is a standardized measure of the strength and direction of the linear relationship between two variables.  
  
> Note that correlation between returns of two stocks may not be an indicator of correlaton between their prices. 

The function `calculate_cov` is defined to calculate each of these metrics: 


```python
def calculate_cov(tickers, start_date, end_date=None, trading_days_in_year=250, print_results=False):
    """
    Calculate variance, covariance, and correlation matrices of selected stocks.

    Parameters:
    -----------
    - tickers (list): List of stock tickers to analyze.
    - start_date (datetime): Start date for historical data retrieval.
    - end_date (datetime): End date for historical data retrieval. Default is None, indicating the current date and time.
    - trading_days_in_year (int): Number of trading days in a year (default is 250).
    - print_results (bool): If True, prints variance information; otherwise, suppresses printing (default is False).
    
    Returns:
    -------
    Pandas Series, Pandas DataFrame, Pandas DataFrame
    """
    # Use current date and time if end_date is not provided
    end_date = end_date or datetime.datetime.now()
    
    # Calculate simple returns of the stocks
    try:
        simple_returns = calculate_returns(tickers, start_date, end_date)
    except Exception as e:
        print(f"Error calculating returns: {e}")
        return

    # Calculate annualized variance and covariance and correlation matrices
    var_annualized = simple_returns.var() * trading_days_in_year
    cov_matrix_annualized = simple_returns.cov() * trading_days_in_year
    corr_matrix = simple_returns.corr()

    # Print outputs if print_info is True
    if print_results:
        for t in tickers:
            print(f'The variance in returns of {t} over the given period is {var_annualized[t]:.2%}')

        print(f'\nHere are their covariance and correlation matrices:')
        print('\nCovariance Matrix:')
        print(cov_matrix_annualized)
        print('\nCorrelation Matrix:')
        print(corr_matrix)

    return var_annualized, cov_matrix_annualized, corr_matrix
```


```python
# Example Usage:
tickers = ['TSLA', 'TM']
start_date = datetime.datetime(2014, 1, 1)
calculate_cov(tickers, start_date, print_results=True)
```

## <u> Portfolio Analysis

Portfolio variance is calculated by the following formula:
$$\mathbf{Portfolio Variance} = \mathbf{w}^T \cdot (\mathbf{C} \cdot \mathbf{w})$$

- Where $\mathbf{w}$ represents the vector of portfolio weights, and $\mathbf{C}$ is the covariance matrix of the asset returns.
- $\mathbf{C} \cdot \mathbf{w}$ computes the product of the covariance matrix and the weights vector.
- The result is then multiplied by the transpose of the weights vector: $\mathbf{w}^T$.
- The final dot product yields the portfolio variance.

This formula is fundamental in modern portfolio theory, helping to quantify the risk associated with a given portfolio of assets, considering both the individual asset volatilities and their pairwise correlations.



```python
weights = np.array([0.5,0.5])
tickers = ['PFE', 'JNJ']
start_date = datetime.datetime(2007,1,1)
end_date = datetime.datetime(2019,2,6)

# Calculate annualized covariance matrix (cov_matrix_annualized) for the selected stocks
# Note: The calculate_cov function also returns the variance and correlation matrix, which are ignored by using the variable name "_"
_, cov_matrix_annualized, _ = calculate_cov(tickers, start_date, print_results=True)

pfolio_var = np.dot(weights.T, np.dot(cov_matrix_annualized, weights))
pfolio_vol = pfolio_var ** 0.5

print(f'Variance of the portflio in the given period is {pfolio_var*100:.3}% \nVolatility of the portfolio is {pfolio_vol*100:.3}%')
```

### <u> Portfolio risk decomposition:

Involves seperating the total risk into the systematic risk and the diversifiable unsystematic risk (also called residual risk). 

The formula for unsystematic risk involves the weights of individual securities and the variance of their returns:

   $$ \text{Unsystematic Risk} = \sum_{i=1}^{n} w_i^2 \cdot \text{Var}(R_i) $$
   
The diverisifiable risk is the residual variance after subtracting the unsystematic variance from the total portfolio variance.





```python
def calculate_portfolio_risk(tickers, weights, start_date, end_date=None, trading_days_in_year=250, print_results=False):
    """
    Calculate the portfolio's total variance, diversifiable variance, and non-diversifiable variance.

    Parameters:
    -----------
    - tickers (list): List of stock tickers to analyze.
    - weights (numpy array): Weights of each stock in the portfolio.
    - start_date (datetime): Start date for historical data retrieval.
    - end_date (datetime): End date for historical data retrieval. Default is None, indicating the current date and time.
    - trading_days_in_year (int): Number of trading days in a year (default is 250).
    - print_results (bool): If True, prints variance information; otherwise, suppresses printing (default is False).
    
    Returns:
    -------
    Pandas Series 
    """
    # Use current date and time if end_date is not provided
    end_date = end_date or datetime.datetime.now()
    
    # Return covariance matrix
    try:
        var_annualized, cov_matrix_annualized, _ = calculate_cov(tickers, start_date)
    except Exception as e:
        raise RuntimeError(f"Error calculating covariance: {e}")

    # Validate lengths of tickers and weights
    if len(tickers) != len(weights):
        raise ValueError('The number of tickers and weights must be the same.')

    # Validate sum of weights
    if abs(sum(weights) - 1) > 1e-10:
        raise ValueError('The sum of weights should be approximately equal to 1.')

    # Calculate portfolio variance
    pfolio_var = np.dot(weights.T, np.dot(cov_matrix_annualized, weights))

    # Calculate the sum of weighted variances of each stock
    sum_weighted_variances = sum(var_annualized * (weights ** 2))
      
    # Calculate diversifiable risk           
    diversifiable_risk = pfolio_var - sum_weighted_variances
    
    # Calculate non-diversifiable risk           
    non_diversifiable_risk = pfolio_var - diversifiable_risk 
    
    if print_results:
        print(f'Total portfolio variance: {pfolio_var*100:.3}% \nDiversifiable variance: {diversifiable_risk*100:.3}% \nNon-diversifiable variance: {non_diversifiable_risk*100:.3}%')
    
    return pfolio_var, diversifiable_risk, non_diversifiable_risk 
```


```python
# Example Usage:
weights = np.array([0.2,0.2,0.2,0.2,0.2])
tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'PFE']
start_date = datetime.datetime(2007,1,1)
end_date = datetime.datetime(2019,2,6)
calculate_portfolio_risk(tickers, weights, start_date, end_date, print_results=True)
```

# <br><br><br><br><br><br><br><u> **Additional Notes**

## 1. Statistics

### **(a) Variance and Standard Deviation:** 

The standard deviation formula involves calculating the squared difference between each value and the mean, summing those squared differences, dividing by the number of values, and then taking the square root:

1. Calculate the mean:  

    Mean = $\mathbf{2 + 4 + 4 + 4 + 5}{5} = 3.8$

2. Calculate the squared differences from the mean:
   > $\mathbf{(2 - 3.8)^2}$  
       $\mathbf{(4 - 3.8)^2}$  
           $\mathbf{(4 - 3.8)^2}$  
               $\mathbf{(4 - 3.8)^2}$  
                   $\mathbf{(5 - 3.8)^2}$  
3. Sum the squared differences:
   $\mathbf{2.4^2 + 0.16^2 + 0.16^2 + 0.16^2 + 1.2^2 = 8}$

4. Divide by the number of values (5):
  Variance = $\frac{8}{5} = 1.6 $

5. Take the square root:
  Standard Deviation = $\sqrt{1.6} = 1.264$


- In a way, variance is the average of the sum of squared deviation of each value from mean. SD is its square root.

$$ \text{variance s}^2 = \frac{\sum\limits_{i=1}^n (x_i - \bar{x})^2}{n-1}$$

- The use of squared differences in the variance calculation serves several purposes:

  1. **Preservation of Sign:**

      Squaring ensures that all differences are positive. This is important because, in a simple mean deviation calculation (without squaring), values above the mean would offset those below the mean, leading to a sum of deviations that might be close to zero.

  2. **Amplifying Larger Deviations:**

      Squaring amplifies larger deviations from the mean. This is desirable because it gives more weight to data points that are further from the mean, reflecting their greater impact on the overall variability of the dataset.

  3. **Mathematical Convenience:**

      Squaring simplifies the mathematical operations involved in calculating the standard deviation. It eliminates the need to deal with negative values and provides a continuous, differentiable function, making it easier to work with mathematically.
      
#### Standard deviation over Variance

While variance provides a measure of how much individual values in a dataset deviate from the mean, to assess the risk associated with a stock, it is generally more intuitive to look at the **standard deviation** rather than the **variance**. Standard deviation is often preferred for several reasons:

1. <u>Units of Measurement:

  - Variance is measured in the squared units of the original data, making its interpretation challenging in the context of the original data.
  - Standard deviation, being the square root of variance, is expressed in the same units as the original data, making it more interpretable and easier to relate to the scale of the data.

2. <u>Interpretability:

  - Standard deviation provides a more intuitive measure of the spread or dispersion in the data compared to the squared units of variance.
  - It represents the average distance of data points from the mean in the same units as the data.

3. <u>Outliers:

  - Squaring in the variance calculation gives more weight to larger deviations from the mean, which can be influenced heavily by outliers.
  - The square root operation in the standard deviation reduces the impact of extreme values, providing a more robust measure of dispersion.

4. <u>Consistency in Reporting:

  - Standard deviation is often used in statistical analyses and reporting, making it a more consistent measure across various contexts.
  - Many statistical methods and models assume normality and use standard deviation as a key parameter.

###  **(b) Variance, Covariance and Correlation:**

**Variance** <br>
- Variance is a statistical measure that quantifies the dispersion or spread of a set of values in a dataset. 
- It calculates the average squared difference between each data point and the mean of the dataset. 
- A high variance indicates that the values are more spread out, while a low variance suggests that the values are closely clustered around the mean.


**Covariance** <br>
- Covariance measures the degree to which two random variables change together. 
- It indicates whether an increase or decrease in one variable is associated with a similar change in the other. 
- A positive covariance signifies a positive relationship (both variables tend to increase or decrease together), while a negative covariance suggests an inverse relationship.
$$\text{cov}(x, y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1}$$


**Correlation** <br>
- Correlation is a standardized measure of the strength and direction of the linear relationship between two variables. 
- It ranges from -1 to 1, where -1 indicates a perfect negative linear relationship, 1 indicates a perfect positive linear relationship, and 0 indicates no linear relationship. 
- Correlation is unitless and is more interpretable than covariance since it is scaled.
$$\rho_{x,y} = \frac{cov(x,y)}{\sigma_x \sigma_y}$$


##### <u> How do they differ?
1. Scale
  - Variance is measured in the squared units of the original data.
  - Covariance is measured in the product of the units of the two variables.
  - Correlation is unitless and ranges from -1 to 1.
2. Interpretability
  - Variance represents the spread of a single variable.
  - Covariance represents the joint variability of two variables but doesn't have a standardized scale.
  - Correlation is a standardized measure, making it easier to interpret and compare the strength and direction of relationships.
3. Range:
  - Variance can take any non-negative value.
  - Covariance can range from negative infinity to positive infinity.
  - Correlation is restricted to the range [-1, 1].

#### **Covariance Matrix:**

- A covariance matrix is a square matrix that summarizes the covariances between multiple variables. 
- In statistics, it is used to describe the joint variability of two or more random variables. 
- The covariance matrix is symmetric, with variances along the diagonal and covariances in the off-diagonal elements.

For a set of n random variables X1, X2, ..., Xn, the covariance matrix Σ is an n × n matrix with elements defined as follows:


$$
\begin{matrix}
\text{Cov}(X_1, X_1) & \text{Cov}(X_1, X_2) & \ldots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Cov}(X_2, X_2) & \ldots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \ldots & \text{Cov}(X_n, X_n)
\end{matrix}
$$





### **(c) For a portfolio:**

- The variance $\sigma^2$ of a portfolio comprising $ n $ assets with respective weights $w_i$ , variances $\sigma_i^2$, and covariances $\text{Cov}(R_i, R_j)$ is given by:

$$\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i \cdot w_j \cdot \text{Cov}(R_i, R_j) $$

- The covariance between two assets $ i $ and $ j $ in a portfolio is given by:

$$ \text{Cov}(R_i, R_j) = \rho_{ij} \cdot \sigma_i \cdot \sigma_j $$
<blockquote>
Where: <br>
- ρ_ij is the correlation coefficient between assets i and j. <br> 
- σ_i and σ_j are the standard deviations of the returns of assets i and j.
</blockquote>
    
- The correlation coefficient $( \rho_p $) between two assets $ i $ and $ j $ in a portfolio is given by:

$$ \rho_p = \frac{\text{Cov}(R_i, R_j)}{\sigma_i \cdot \sigma_j} $$

These formulas are crucial for understanding the risk and return characteristics of a portfolio with multiple assets. These formulas assume a well-diversified portfolio, and the weights of the assets in the portfolio sum up to 1 $( \sum_{i=1}^{n} w_i = 1 $).


## 2. Finance

#### <u>**Portfolio Risk Decomposition:**

Involves separating the total risk into the non-diversifiable systematic risk and the diversifiable unsystematic risk. It is given by:
> **Total Portfolio Variance** = Systematic Risk + Unsystematic Risk <br>
where: <br>
  > **Systematic Risk** = Weighted Beta × Market Variance <br>
  > **Unsystematic Risk** = Total Portfolio Variance − Sum of individual weighted variances
    
    
    
1. **Systematic Risk (Non-Diversifiable/Market Risk):**
   - This component is related to factors that affect the entire market, such as economic conditions, interest rates, and geopolitical events.
   - It represents the risk that cannot be diversified away, such as recession, low consumer spending, wars, forces of nature etc.
   - The primary measure of systematic risk is beta ($\beta$), which represents the sensitivity of an individual security's returns to the overall market returns.
    - The formula for systematic risk involves the beta of each security and the variance of the market returns:

   $$\mathbf{\text{Systematic Risk} = \sum_{i=1}^{n} w_i \cdot \beta \cdot\text{Var}(M) }$$


    <blockquote>
   Where: <br>
   - $w_i$ is the weight of security $i$ in the portfolio. <br>
   - $\text{Var}(R_i)$ is the variance of the returns of security $i$. <br> 
   - $\beta_i$ is the beta of security $i$.
                                                                                          </blockquote>

2. **Unsystematic Risk (Diversifiable/Company specific/idiosyncratic risk):**
   - Unsystematic risk is related to specific characteristics of individual securities and can be diversified away by holding a well-diversified portfolio.    
   - The primary measure of unsystematic risk is the variance of the returns of individual securities.
   - Diversification helps to reduce the impact of individual variances on the overall portfolio.
   - The formula for unsystematic risk involves the weights of individual securities and the variance of their returns:

   $$\mathbf{\text{Unsystematic Risk} = \sum_{i=1}^{n} w_i^2 \cdot \text{Var}(R_i) }$$

    <blockquote>
   Where: <br>
   - $w_i$ is the weight of security $i$ in the portfolio. <br>
   - $\text{Var}(R_i)$ is the variance of the returns of security $i$.
                                                                                          </blockquote>
    
#### **Total Portfolio Risk:**
$$ \text{Total Portfolio Risk} = \text{Systematic Risk} + \text{Unsystematic Risk} $$

This decomposition helps investors and portfolio managers understand the sources of risk in a portfolio and the potential benefits of diversification in reducing overall risk.

#### <u>Some additional notes about unsystematic risk:
It can be described as the sum of two components: 
1. the variances of the individual securities within the portfolio and 
2. the covariances between the securities. 


The variances capture the individual risk associated with each security, while the covariances represent the interdependence or co-movements between pairs of securities. 
Together, these components contribute to the overall unsystematic risk of the portfolio. <br> <br>
While variances can be mitigated through diversification, the covariances, when diversified, help reduce the impact of specific securities on the overall portfolio risk.  
  - Diversifying a portfolio with assets that have low or negative covariances (e.g. stocks from different countries) can help reduce the overall risk. When securities have low covariances, they may not move in the same direction or by the same magnitude, providing a form of risk reduction. 
  - Poor performance of one security is less likely to coincide with the poor performance of another, thereby reducing the overall impact on the portfolio.
  
  
  
On the other hand, regardless of the number of securities contained in a portfolio, their variances and covariances, systematic risk will continue to exist.

## 3. Python

### (a) NumPy np.dot() function

The np.dot function in NumPy is a versatile tool for matrix multiplication and array operations. Its behavior depends on the type and dimensions of the input arrays. 
1. <u> 1-D Arrays (Vectors): 
  
    If both arrays are 1-D, the dot product is the inner product of vectors (scalar result). 
    Example: np.dot([a, b], [c, d]) results in a⋅c+b⋅d. 

2. <u> 2-D Arrays (Matrices): 
  
    If both arrays are 2-D matrices, it performs matrix multiplication. 

3. <u> Combining 1-D and 2-D Arrays: 
  
    If one of the arrays is 1-D and the other is 2-D (matrix), it behaves as if the 1-D array is a row vector, i.e. it gets tranposed to give a 1x2 matrix for matrix multiplication. Thus a row vector gets tranposed to column vector and vice versa. <br>
    Example: For a 1-D array [a, b] and a 2-D array [[c, d], [e, f]], np.dot() results in a 1-D array. 

4. <u> Transpose Operation: 
    
    The dot product can also be achieved by transposing one of the arrays.<br>
    Example: np.dot(A, B.T) is equivalent to the dot product of A and the transpose of B. 

5. <u> Special Cases (Scalars): 
    
    If both arguments are scalars, the dot product is equivalent to multiplication. 
    Example: np.dot(2, 3) results in 2 ⋅ 3  

### (b) Pandas DataFrame

The syntax for creating a DataFrame in pandas involves using the pd.DataFrame() constructor. You can pass a dictionary, list of dictionaries, or other iterable objects to specify the data and columns.

- Here's a basic example using a dictionary:


```python
import pandas as pd

# Example data
data = {
    'Column1': [1, 2, 3],
    'Column2': ['A', 'B', 'C']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

```

- You can also create an empty DataFrame and then append rows or columns to it as needed. The following example demonstrates this:


```python
import pandas as pd

# Create an empty DataFrame
df = pd.DataFrame()

# Add columns
df['Column1'] = [1, 2, 3]
df['Column2'] = ['A', 'B', 'C']

# Display the DataFrame
print(df)

```

### (c) Ignoring function results


```python
var_annualized, _, _ = calculate_cov(tickers, start_date)
```

var_annualized is a variable, and the underscores (_) are used as placeholder variables to indicate that you're intentionally discarding the other elements of the tuple returned by calculate_cov.

var_annualized: This variable stores the value of the variance in returns, which is the first element of the tuple returned by calculate_cov.  

_: The underscores are conventionally used as placeholders to indicate that the corresponding values (covariance matrix and correlation matrix in this case) are being ignored and not assigned to variables.
So, var_annualized is a new variable that holds the value you're interested in, and the underscores are used to "throw away" the other values returned by the function.

### (d) Error handling

Error handling in Python involves using try, except, and optionally finally blocks to manage exceptions that might occur during the execution of a program:

- **try** Block:
The try block contains the code where an exception might occur. Python attempts to execute the code within this block.

- **except** Block:
If an exception occurs in the try block, the corresponding except block is executed. You can have multiple except blocks to catch different types of exceptions or handle them in different ways.

- **finally** Block (Optional):
The finally block, if present, is executed whether an exception occurs or not. It is typically used for cleanup actions.




```python
def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError as zde:
        print(f"Error: {zde}. Cannot divide by zero.")
        result = None
    except TypeError as te:
        print(f"Error: {te}. Please provide valid numbers.")
        result = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        result = None
    finally:
        print("The output is:")
        # This will always be executed.

    return result

# Example usage:
result = divide_numbers(10, 2)
print(result)


```

### (d) raise statement

- Raises an exception, interrupting the normal flow of the program.
- Provides a structured way to handle errors, allowing to specify the type of exception and include a custom error message.


```python
try:
    # Some code that may raise an exception
    result = 1 / 0
except ZeroDivisionError as e:
    # Handle the specific exception (division by zero)
    raise RuntimeError("Custom error message") from e

```

### (e) runtime error

A runtime error, also known as an exception, occurs during the execution of a program. These errors can happen for various reasons, such as invalid input, division by zero, or attempting to access a variable or element that doesn't exist. Runtime errors are typically detected while the program is running.

Here are some common types of runtime errors in Python:

- SyntaxError: Raised when the Python interpreter encounters a syntax error, indicating a mistake in the program's structure. This error prevents the program from running.
- NameError: Raised when a local or global name is not found. This occurs when trying to access a variable or function that has not been defined.
- TypeError: Raised when an operation or function is applied to an object of the wrong type.
- ValueError: Raised when a function receives an argument of the correct type but an inappropriate value.
- ZeroDivisionError: Raised when attempting to divide a number by zero.
- IndexError: Raised when trying to access an index that does not exist in a list or tuple.
- FileNotFoundError: Raised when attempting to open or manipulate a file that does not exist.
- AttributeError: Raised when attempting to access an attribute or method that is not defined for an object.
- KeyError: Raised when trying to access a dictionary key that does not exist.
- RuntimeError: This is a generic error that can be raised for various reasons during runtime.
>Example:
raise RuntimeError("Custom runtime error")
