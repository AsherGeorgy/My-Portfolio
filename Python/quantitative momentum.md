# Momentum-Based Investment Strategy

# Table of Contents

1. [Overview](#overview)
2. [Data Collection](#1-data-collection)
    - [Fetching Tickers](#fetching-tickers)
    - [Downloading Stock Data](#downloading-stock-data)
3. [Stock Prices and Returns Calculation](#2-stock-prices-and-returns-calculation)
4. [Top 50 Stocks](#3-top-50-stocks)
5. [Portfolio Allocation](#4-portfolio-allocation)
    - [User Input for Portfolio Size](#user-input-for-portfolio-size)
    - [Calculating Shares to Buy](#calculating-shares-to-buy)
6. [Momentum Metrics Calculation](#5-momentum-metrics-calculation)
    - [Combining DataFrames](#combining-dataframes)
7. [HQM Score Calculation](#6-hqm-score-calculation)
    - [Reordering Columns](#reordering-columns)
8. [Share Allocation](#7-share-allocation)
9. [Export](#8-export)
10. [Conclusion](#conclusion)
11. [Notes](#notes)
    - [Position Size](#1-position-size)
    - [Number of Shares to Buy](#2-number-of-shares-to-buy)
    - [HQM Score](#3-hqm-score)

## Overview

This project outlines a Python-based strategy to analyze S&P 500 companies based on their momentum. The strategy involves fetching stock data, calculating returns, determining portfolio allocation, and computing an HQM (High-Quality Momentum) Score.

---

## 1. Data Collection

#### Fetching Tickers

The S&P500 company tickers are fetched from Wikipedia:


```python
import pandas as pd
import yfinance as yf

link = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

sp500_df = pd.DataFrame()
sp500_df['Ticker'] = pd.read_html(link)[0]['Symbol']
```


```python
print(sp500_df.head())
```

#### Output

| **Ticker** |
|------------|
| MMM        |
| AOS        |
| ABT        |
| ABBV       |
| ACN        |
    

#### Downloading Stock Data

Using the `yfinance` library, the adjusted closing prices for the last year are downloaded:


```python
tickers_list = sp500_df['Ticker'].tolist()

try:
    data_download = yf.download(tickers_list, period ='1y')['Adj Close']
except Exception as e:
    print(f"Error downloading data: {e}")
    data_download = pd.DataFrame()

```

```python
print(data_download.head())
```

#### Output

| Ticker       | A         | AAPL       | ABBV       | ABNB       | ABT        | ... | ZBH        | ZBRA       | ZTS        |
|--------------|-----------|------------|------------|------------|------------|-----|------------|------------|------------|
| **Date**     |           |            |            |            |            | ... |            |            |            |
| 2023-10-10   | 112.526970| 177.481812 | 143.351410 | 131.589996 | 95.572906  | ... | 111.261375 | 222.399994 | 173.903336 |
| 2023-10-11   | 112.616356| 178.884628 | 143.784668 | 130.000000 | 90.753120  | ... | 104.477989 | 217.160004 | 175.210358 |
| 2023-10-12   | 108.206276| 179.789993 | 144.212555 | 125.970001 | 88.842117  | ... | 101.869759 | 210.869995 | 172.596298 |
| 2023-10-13   | 109.765694| 177.939468 | 143.881927 | 124.080002 | 89.511955  | ... | 103.436676 | 206.919998 | 172.665619 |
| 2023-10-16   | 110.749023| 177.810104 | 143.172028 | 125.190002 | 90.802376  | ... | 104.180481 | 214.970001 | 173.457748 |

[5 rows x 503 columns]
    

## 2. Stock Prices and Returns Calculation

The current stock prices and one-year returns are calculated for each ticker:


```python
sp500_df['Stock Price'] = None
sp500_df['One Year Return'] = None
sp500_df

for t in tickers_list:
    try:
        # To avoid errors
        if t in data_download.columns:
            # Assigning close prices in Stock Price column to the corresponding ticker   
            # in the Ticker column. 
            sp500_df.loc[sp500_df['Ticker'] == t,'Stock Price'] = data_download[t].iloc[-1]
    
            # Calculate one-year return and doing the same
            one_year_return = ((data_download[t].iloc[-1]/data_download[t].iloc[0]) - 1) * 100
            sp500_df.loc[sp500_df['Ticker'] == t, 'One Year Return'] = one_year_return
        else:
            # Assign None to any ticker in ticker's list that isn't in the downloaded 
            # close price dataframe. 
            sp500_df.loc[sp500_df['Ticker'] == t, 'Stock Price'] = None
            sp500_df.loc[sp500_df['Ticker'] == t, 'One Year Return'] = None
    except Exception as e:
        print(f"Error downloading data for ticker {t}: {e}")
```


```python
print(sp500_df.head())
```

#### Output

| Ticker | Stock Price | One Year Return |
|--------|-------------|------------------|
| MMM    | 135.009995  | 86.231578        |
| AOS    | 87.68       | 28.670969        |
| ABT    | 115.949997  | 21.320991        |
| ABBV   | 194.75      | 35.85496         |
| ACN    | 365.079987  | 18.781146        |

    

## 3. Top 50 Stocks

The best 50 best performing S&P500 stocks are selected for further analysis:


```python
# Selecting the top 50 best performing stocks
sp500_df.sort_values('One Year Return', ascending=False, inplace=True)
# Slicing syntax df[start:end]
sp500_df = sp500_df[:50].copy()

# Drop the 'index' column and reset index
sp500_df.reset_index(inplace=True, drop=True)
```


```python
print(sp500_df)
```

#### Output

| Ticker | Stock Price | One Year Return |
|--------|-------------|------------------|
| VST    | 124.18      | 292.337586       |
| NVDA   | 132.649994  | 189.729104       |
| PLTR   | 43.130001   | 142.303387       |
| CEG    | 262.279999  | 133.892372       |
| NRG    | 88.559998   | 128.381874       |
| FICO   | 2024.98999  | 127.744468       |
| HWM    | 103.699997  | 123.955314       |
| AVGO   | 185.949997  | 119.959293       |
| KKR    | 134.949997  | 119.043857       |
| RCL    | 193.029999  | 116.591403       |
| GE     | 189.279999  | 113.934371       |
| GDDY   | 161.210007  | 113.099806       |
| ANET   | 406.929993  | 108.971397       |
| IRM    | 120.489998  | 108.315614       |
| TRGP   | 163.399994  | 101.121754       |
| AXON   | 432.140015  | 100.724617       |
| NFLX   | 727.429993  | 94.854275        |
| TT     | 399.720001  | 93.870389        |
| MPWR   | 936.01001   | 92.238296        |
| MHK    | 159.0       | 91.497052        |
| PHM    | 139.389999  | 89.845527        |
| DECK   | 160.440002  | 89.291129        |
| MMM    | 135.009995  | 86.231578        |
| META   | 590.51001   | 84.031815        |
| URI    | 813.51001   | 83.15711        |
| DELL   | 122.339996  | 83.129183        |
| AXP    | 271.420013  | 81.985022        |
| BK     | 74.019997   | 80.787239        |
| LDOS   | 167.669998  | 79.886842        |
| SYF    | 52.0        | 78.937473        |
| RL     | 195.729996  | 77.5965         |
| PGR    | 250.759995  | 77.405703       |
| ERIE   | 532.599976  | 77.324459       |
| FITB   | 42.630001   | 77.240827       |
| WAB    | 184.220001  | 76.504901       |
| TDG    | 1386.959961 | 75.274368       |
| DVA    | 159.300003  | 74.517972       |
| DHI    | 185.259995  | 74.234197       |
| PWR    | 307.609985  | 73.419606       |
| RTX    | 123.949997  | 73.303954       |
| IBM    | 234.300003  | 71.418886       |
| KLAC   | 804.630005  | 71.208056       |
| KEY    | 16.870001   | 70.929065       |
| NTAP   | 127.269997  | 70.632306       |
| CTAS   | 209.139999  | 69.973331       |
| STX    | 109.290001  | 69.118534       |
| NOW    | 938.650024  | 68.126463       |
| SPG    | 169.229996  | 68.110253       |
| ZBRA   | 373.5       | 67.940652       |
| TPR    | 44.810001   | 67.811928       |

    

## 4. Portfolio Allocation 

#### User Input for Portfolio Size
The following function prompts the user to input their portfolio size:



```python
def enter_portfolio():
    while True:
        portfolio_size = input("Enter portfolio size: ")
        try:
            portfolio_size = float(portfolio_size)
            return portfolio_size
        except ValueError:
            print(f"The entered value '{portfolio_size}' is not a valid portfolio size. Please enter an integer or float and try again.\n")

portfolio_size = enter_portfolio()
```

    Enter portfolio size:  10000000
    

#### Calculating Shares to Buy

The Position Size and Number of shares to buy are calculated:


```python
import math

position_size = portfolio_size/len(sp500_df.index)
sp500_df['No of shares to buy'] = (position_size/sp500_df['Stock Price']).apply(math.floor)
```


```python
print(sp500_df.head())
```

#### Output

| Ticker | Stock Price | One Year Return | No of shares to buy |
|--------|-------------|------------------|----------------------|
| VST    | 124.18      | 292.337586       | 1610                 |
| NVDA   | 132.649994  | 189.729104       | 1507                 |
| PLTR   | 43.130001   | 142.303387       | 4637                 |
| CEG    | 262.279999  | 133.892372       | 762                  |
| NRG    | 88.559998   | 128.381874       | 2258                 |

    

## 5. Momentum Metrics Calculation

For each stock, various momentum metrics are calculated:


```python
# Initiate an empty list to hold the dictionaries for each ticker
# This list is later converted to the hqm dataframe
data_frames = []

# Take each ticker and populate all required columns
for t in sp500_df['Ticker']:
    if t in data_download.columns:
        # Temporary dataframe for current ticker to calculate returns
        temp_df = data_download[t].dropna()
        
        # Calculate returns
        one_month_return = temp_df.pct_change(periods=21).iloc[-1] * 100
        three_month_return = temp_df.pct_change(periods=63).iloc[-1] * 100
        six_month_return = temp_df.pct_change(periods=126).iloc[-1] * 100

        # Dataframe for current ticker to populate all required columns
        ticker_df = pd.DataFrame([{
            'Ticker':t,
            'Stock Price':sp500_df.loc[sp500_df['Ticker'] == t, 'Stock Price'].iloc[0],
            'No of shares to buy': None,
            'One Year Price Return':sp500_df.loc[sp500_df['Ticker'] == t, 'One Year Return'].iloc[0],
            'Six Month Price Return':six_month_return,
            'Three Month Price Return':three_month_return,
            'One Month Price Return': one_month_return,
            'HQM Score': None
        }])

        # Append the current ticker dataframe to the initiated list
        data_frames.append(ticker_df)
    else:
        print(f"Ticker {t} not found in price data.")
```

#### Combining DataFrames


```python
# Vertically stack each ticker_df appended to a single dataframe
# Each ticker_df forms a new row
hqm_df = pd.concat(data_frames, ignore_index=True) 
print(hqm_df.head())
```

#### Output

| Ticker | Stock Price | No of shares to buy | One Year Price Return | Six Month Price Return | Three Month Price Return | One Month Price Return | HQM Score |
|--------|-------------|----------------------|-----------------------|------------------------|--------------------------|-----------------------|-----------|
| VST    | 124.180000  | None                 | 292.337586            | 76.369412              | 35.875208                | 63.611046             | None      |
| NVDA   | 132.649994  | None                 | 189.729104            | 52.428496              | 4.129778                 | 22.720947             | None      |
| PLTR   | 43.130001   | None                 | 142.303387            | 92.372885              | 56.041975                | 24.079411             | None      |
| CEG    | 262.279999  | None                 | 133.892372            | 37.727767              | 21.180846                | 45.509015             | None      |
| NRG    | 88.559998   | None                 | 128.381874            | 22.352141              | 11.821307                | 13.625859             | None      |

    

## 6. HQM Score Calculation

The percentiles of returns from each period are calculated and inserted for each Ticker.


```python
def calculate_percentiles(df, column_name):
    return df[column_name].rank(pct=True) * 100

# Compute percentiles for each return period
hqm_df['One Year Return Percentile'] = calculate_percentiles(hqm_df, 'One Year Price Return')
hqm_df['Six Month Return Percentile'] = calculate_percentiles(hqm_df, 'Six Month Price Return')
hqm_df['Three Month Return Percentile'] = calculate_percentiles(hqm_df, 'Three Month Price Return')
hqm_df['One Month Return Percentile'] = calculate_percentiles(hqm_df, 'One Month Price Return')
```

#### Reordering Columns


```python
# The columns are reordered for clarity
hqm_df = hqm_df[['Ticker','Stock Price','HQM Score','No of shares to buy','One Year Price Return','One Year Return Percentile','Six Month Price Return','Six Month Return Percentile','Three Month Price Return','Three Month Return Percentile','One Month Price Return','One Month Return Percentile']]
```

The HQM Score of each ticker is the mean of its respective percentiles. 

#### Calculating the HQM Score


```python
from statistics import mean

time_periods = [
    'One Year',
    'Six Month',
    'Three Month',
    'One Month'
]

# Populate HQM score for each ticker row
for row in hqm_df.index:
    # Collect return percentile for each time period for each ticker row. 
    # HQM score is the mean of these percentiles. 
    momentum_percentiles = []
    for t in time_periods:
        momentum_percentiles.append(hqm_df.loc[row,f'{t} Return Percentile'])
    hqm_df.loc[row,'HQM Score'] = mean(momentum_percentiles)

hqm_df = hqm_df.sort_values(by='HQM Score', ascending=False)
print(hqm_df.head())
```

#### Output

| Ticker | Stock Price  | HQM Score | No of shares to buy | One Year Price Return | One Year Return Percentile | Six Month Price Return | Six Month Return Percentile | Three Month Price Return | Three Month Return Percentile | One Month Price Return | One Month Return Percentile |
|--------|--------------|-----------|----------------------|-----------------------|----------------------------|------------------------|-----------------------------|--------------------------|-------------------------------|-----------------------|-----------------------------|
| VST    | 124.180000   | 98.0      | None                 | 292.337586            | 100.0                      | 76.369412              | 98.0                        | 35.875208                | 94.0                          | 63.611046             | 100.0                      |
| PLTR   | 43.130001    | 97.0      | None                 | 142.303387            | 96.0                       | 92.372885              | 100.0                       | 56.041975                | 100.0                         | 24.079411             | 92.0                       |
| CEG    | 262.279999   | 83.0      | None                 | 133.892372            | 94.0                       | 37.727767              | 76.0                        | 21.180846                | 64.0                          | 45.509015             | 98.0                       |
| FICO   | 2024.989990  | 82.0      | None                 | 127.744468            | 90.0                       | 71.430629              | 96.0                        | 29.184314                | 84.0                          | 12.669699             | 58.0                       |
| AXON   | 432.140015   | 80.5      | None                 | 100.724617            | 70.0                       | 35.782065              | 70.0                        | 48.420116                | 98.0                          | 20.256020             | 84.0                       |

    

## 7. Share allocation

The number of shares to buy for each stock is recalculated:


```python
position_size = portfolio_size/len(hqm_df.index)
hqm_df['No of shares to buy'] = (position_size/hqm_df['Stock Price']).apply(math.floor)
```


```python
print(hqm_df)
```

#### Output

|Ticker|Stock Price|HQM Score|No of shares to buy|One Year Price Return|One Year Return Percentile |Six Month Price Return|Six Month Return Percentile |Three Month Price Return|Three Month Return Percentile |One Month Price Return|One Month Return Percentile|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|VST|124.18|98|1610|292.34|100|76.369412|98|35.875208|94|63.611046|100|
|PLTR|43.13|97|4637|142.3|96|92.372885|100|56.041975|100|24.079411|92|
|CEG|262.28|83|762|133.89|94|37.727767|76|21.180846|64|45.509015|98|
|FICO|2024.99|82|98|127.74|90|71.430629|96|29.184314|84|12.669699|58|
|AXON|432.14|80.5|462|100.72|70|35.782065|70|48.420116|98|20.25602|84|
|RCL|193.03|79.5|1036|116.59|82|47.808349|88|18.569138|62|22.224321|86|
|HWM|103.7|79|1928|123.96|88|59.163307|92|30.291591|86|11.325813|50|
|KKR|134.95|77.5|1482|119.04|84|36.74325|74|24.168774|78|14.306285|74|
|TRGP|163.4|73.5|1223|101.12|72|44.210081|84|22.983857|74|13.330561|64|
|NVDA|132.65|72|1507|189.73|98|52.428496|90|4.129778|12|22.720947|88|
|AVGO|185.95|71|1075|119.96|86|41.529206|82|9.359353|20|25.876517|96|
|ANET|406.93|71|491|108.97|76|41.187284|80|13.164993|34|24.538636|94|
|IRM|120.49|68.5|1659|108.32|74|61.358667|94|28.259069|82|6.193473|24|
|TT|399.72|65|500|93.87|66|34.575364|66|16.293704|50|15.482621|78|
|MMM|135.01|61.5|1481|86.23|56|47.555352|86|31.490659|90|3.781996|14|
|IBM|234.3|60.5|853|71.42|20|28.303301|58|32.549633|92|14.11455|72|
|MHK|159|58.5|1257|91.5|62|32.865382|62|31.079964|88|6.099025|22|
|GE|189.28|57.5|1056|113.93|80|21.460298|28|17.070375|56|13.522785|66|
|ERIE|532.6|56|375|77.32|36|36.186808|72|47.282207|96|5.350638|20|
|BK|74.02|55.5|2701|80.79|46|34.920256|68|21.326063|66|9.643007|42|
|URI|813.51|55|245|83.16|52|19.8232|20|22.379193|68|15.543909|80|
|NRG|88.56|54.5|2258|128.38|92|22.352141|34|11.821307|24|13.625859|68|
|MPWR|936.01|53.5|213|92.24|64|40.84593|78|12.795789|28|10.567019|44|
|PHM|139.39|51.5|1434|89.85|60|27.818796|56|22.61888|72|5.191653|18|
|GDDY|161.21|51|1240|113.1|78|28.751703|60|13.432319|36|7.717495|30|
|PWR|307.61|48|650|73.42|24|21.042828|26|16.492338|52|23.066378|90|
|META|590.51|47.5|338|84.03|54|13.818015|6|15.286394|48|17.092919|82|
|AXP|271.42|46|736|81.99|48|25.403078|46|13.980836|44|10.883075|46|
|LDOS|167.67|44|1192|79.89|44|33.519863|64|12.846267|30|9.103281|38|
|SYF|52|42.5|3846|78.94|42|26.884825|54|6.584923|14|12.676053|60|
|WAB|184.22|42.5|1085|76.5|32|25.168793|44|13.462564|38|12.342972|56|
|NOW|938.65|40|213|68.13|8|22.03097|32|25.297008|80|9.126315|40|
|TDG|1386.96|40|144|75.27|30|20.037708|22|16.621971|54|11.718644|54|
|DHI|185.26|38|1079|74.23|26|25.437389|48|23.936021|76|-1.210475|2|
|RL|195.73|35|1021|77.6|40|18.626127|14|6.855806|16|13.810412|70|
|ZBRA|373.5|34|535|67.94|4|24.53736|38|13.760962|42|11.54248|52|
|RTX|123.95|34|1613|73.3|22|23.680898|36|22.506172|70|3.034076|8|
|NFLX|727.43|33.5|274|94.85|68|17.596749|12|11.440826|22|7.988183|32|
|PGR|250.76|33|797|77.41|38|21.745079|30|18.430073|60|0.693848|4|
|DELL|122.34|32.5|1634|83.13|50|-0.103937|2|-12.759853|2|14.722428|76|
|CTAS|209.14|32|956|69.97|12|25.921755|52|17.096939|58|2.466869|6|
|DECK|160.44|31.5|1246|89.29|58|18.825375|16|8.327328|18|8.075587|34|
|FITB|42.63|29|4691|77.24|34|24.818823|40|12.390288|26|4.956123|16|
|NTAP|127.27|27.5|1571|70.63|14|25.072019|42|-2.775283|6|11.169539|48|
|DVA|159.3|26|1255|74.52|28|20.626992|24|13.58289|40|3.744708|12|
|STX|109.29|24|1829|69.12|10|25.87431|50|3.689876|8|7.593704|28|
|KLAC|804.63|23.5|248|71.21|18|17.554765|10|-5.492433|4|12.778569|62|
|KEY|16.87|20.5|11855|70.93|16|16.725788|8|12.865373|32|6.234261|26|
|SPG|169.23|20|1181|68.11|6|19.287957|18|15.165027|46|3.214192|10|
|TPR|44.81|13|4463|67.81|2|9.938648|4|3.938445|10|8.841398|36|


    

## 8. Export 

The final dataframe is exported to Excel using `xlsxwriter`. Download it [here](https://github.com/AsherGeorgy/My-Portfolio/raw/refs/heads/main/Python/assets/momentum_strategy.xlsx)



```python
import xlsxwriter

writer = pd.ExcelWriter('momentum_strategy.xlsx')
hqm_df.to_excel(writer, sheet_name='Momentum Strategy', index=False)
writer.close()
```

## Conclusion
This strategy effectively identifies stocks with strong momentum, allowing investors to make informed decisions.  <br><br>The analysis highlighted three key performers:
- **Vistra Corp (VST)**, with an impressive one-year price return of 292.34% and an HQM score of 98.0, indicating exceptional momentum.
- **Palantir Technologies Inc. (PLTR)**, achieved a one-year return of 142.30% and an HQM score of 97.0
- **Constellation Energy Corp (CEG)**, reported a 133.89% return and an HQM score of 83.0. <br><br>

The detailed results demonstrate the potential for robust portfolio performance by focusing on high-quality momentum stocks. Investors can use this framework to enhance their strategies and optimize their investment decisions based on strong momentum indicators.

---

# Notes

## 1. Position Size

The **position size** represents the amount of money allocated to each stock. This is calculated by dividing the total portfolio size by the number of stocks.

**Mathematical Formula:**

$$ \text{Position Size} = \frac{\text{Total Portfolio Size}}{\text{Number of Stocks}} $$

> <u>Example Calculation</u>:
> <br> Total Portfolio Size: 9,000
> <br> Number of Stocks: 3
> <br> Position Size = 9,000/3 = 3,000 
> <br>Each stock gets approximately \$3,000 to invest.

## 2. Number of Shares to Buy

For each stock, you need to determine how many shares can be bought with the allocated position size. This is done by dividing the position size by the stock price and rounding down to the nearest integer.

**Mathematical Formula:**

$$ \text{Number of Shares to Buy} = \left\lfloor \frac{\text{Position Size}}{\text{Stock Price}} \right\rfloor \text{rounded down to the nearest integer.} $$

This approach ensures that you evenly distribute your investment across the stocks in your portfolio and determine how many whole shares you can buy with the allocated funds.mine how many whole shares you can buy with the allocated funds.
mine how many whole shares you can buy with the allocated funds.

## 3. HQM Score

The **HQM (High-Quality Momentum) Score** is a metric used in quantitative finance to assess the momentum of a financial asset, such as a stock, over various time periods. Momentum is a measure of the tendency of an asset's price to continue moving in its current direction—either up or down—over time.

- **Multiple Time Periods:** The HQM Score aggregates momentum measurements over different time periods, such as one year, six months, three months, and one month. This allows for a more comprehensive view of the asset's performance over time, capturing both short-term and long-term trends.

- **Percentiles:** For each time period, the asset's return is compared to other assets in the same dataset, and a percentile ranking is assigned. For example, if a stock's one-year return is in the 90th percentile, it means it has outperformed 90% of the other assets over that period.

- **Average Percentile:** The HQM Score is typically calculated by taking the average of these percentile rankings across all selected time periods. A higher HQM Score indicates stronger and more consistent momentum across different timeframes.

Investors and analysts use the HQM Score to identify assets that have shown strong performance over multiple time periods, with the expectation that these assets may continue to perform well in the future. It can be particularly useful in momentum-based strategies, where the goal is to invest in assets that are trending upwards.

