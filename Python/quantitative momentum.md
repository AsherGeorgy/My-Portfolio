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

      Ticker
    0    MMM
    1    AOS
    2    ABT
    3   ABBV
    4    ACN
    

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

    [**********************78%%***********           ]  393 of 503 completed

    $BF.B: possibly delisted; No price data found  (period=1y)
    

    [*********************100%%**********************]  503 of 503 completed
    
    4 Failed downloads:
    ['BRK.B']: YFChartError('%ticker%: No data found, symbol may be delisted')
    ['AMTM']: YFInvalidPeriodError("%ticker%: Period '1y' is invalid, must be one of ['1d', '5d', '1mo', 'ytd', 'max']")
    ['SW']: YFInvalidPeriodError("%ticker%: Period '1y' is invalid, must be one of ['1d', '5d', '1mo', '3mo', '6mo', 'ytd', 'max']")
    ['BF.B']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (period=1y)')
    


```python
print(data_download.head())
```

    Ticker               A        AAPL        ABBV        ABNB        ABT  \
    Date                                                                    
    2023-10-10  112.526970  177.481812  143.351410  131.589996  95.572906   
    2023-10-11  112.616356  178.884628  143.784668  130.000000  90.753120   
    2023-10-12  108.206276  179.789993  144.212555  125.970001  88.842117   
    2023-10-13  109.765694  177.939468  143.881927  124.080002  89.511955   
    2023-10-16  110.749023  177.810104  143.172028  125.190002  90.802376   
    
    Ticker           ACGL         ACN        ADBE         ADI        ADM  ...  \
    Date                                                                  ...   
    2023-10-10  81.860001  307.355164  532.719971  172.952454  71.797333  ...   
    2023-10-11  83.019997  308.847321  549.909973  171.920441  71.361610  ...   
    2023-10-12  83.050003  300.763947  559.630005  170.623093  70.461113  ...   
    2023-10-13  84.160004  298.263824  548.760010  168.136475  70.993660  ...   
    2023-10-16  85.699997  301.732361  550.739990  169.787659  72.155586  ...   
    
    Ticker             WTW         WY       WYNN        XEL         XOM  \
    Date                                                                  
    2023-10-10  205.751541  29.237757  92.949081  55.904263  106.708534   
    2023-10-11  207.863739  29.762112  93.680893  56.626411  102.882683   
    2023-10-12  207.616989  29.179493  92.513954  55.990917  102.863358   
    2023-10-13  210.469437  29.072680  89.280174  56.674557  106.148186   
    2023-10-16  212.275665  29.208622  89.319733  57.348568  106.225479   
    
    Ticker            XYL         YUM         ZBH        ZBRA         ZTS  
    Date                                                                   
    2023-10-10  90.903435  118.034874  111.261375  222.399994  173.903336  
    2023-10-11  91.318665  116.083466  104.477989  217.160004  175.210358  
    2023-10-12  90.250954  113.994789  101.869759  210.869995  172.596298  
    2023-10-13  89.143700  114.612564  103.436676  206.919998  172.665619  
    2023-10-16  90.854019  116.152115  104.180481  214.970001  173.457748  
    
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

      Ticker Stock Price One Year Return
    0    MMM  135.009995       86.231578
    1    AOS       87.68       28.670969
    2    ABT  115.949997       21.320991
    3   ABBV      194.75        35.85496
    4    ACN  365.079987       18.781146
    

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

       Ticker  Stock Price One Year Return
    0     VST       124.18      292.337586
    1    NVDA   132.649994      189.729104
    2    PLTR    43.130001      142.303387
    3     CEG   262.279999      133.892372
    4     NRG    88.559998      128.381874
    5    FICO   2024.98999      127.744468
    6     HWM   103.699997      123.955314
    7    AVGO   185.949997      119.959293
    8     KKR   134.949997      119.043857
    9     RCL   193.029999      116.591403
    10     GE   189.279999      113.934371
    11   GDDY   161.210007      113.099806
    12   ANET   406.929993      108.971397
    13    IRM   120.489998      108.315614
    14   TRGP   163.399994      101.121754
    15   AXON   432.140015      100.724617
    16   NFLX   727.429993       94.854275
    17     TT   399.720001       93.870389
    18   MPWR    936.01001       92.238296
    19    MHK        159.0       91.497052
    20    PHM   139.389999       89.845527
    21   DECK   160.440002       89.291129
    22    MMM   135.009995       86.231578
    23   META    590.51001       84.031815
    24    URI    813.51001        83.15711
    25   DELL   122.339996       83.129183
    26    AXP   271.420013       81.985022
    27     BK    74.019997       80.787239
    28   LDOS   167.669998       79.886842
    29    SYF         52.0       78.937473
    30     RL   195.729996         77.5965
    31    PGR   250.759995       77.405703
    32   ERIE   532.599976       77.324459
    33   FITB    42.630001       77.240827
    34    WAB   184.220001       76.504901
    35    TDG  1386.959961       75.274368
    36    DVA   159.300003       74.517972
    37    DHI   185.259995       74.234197
    38    PWR   307.609985       73.419606
    39    RTX   123.949997       73.303954
    40    IBM   234.300003       71.418886
    41   KLAC   804.630005       71.208056
    42    KEY    16.870001       70.929065
    43   NTAP   127.269997       70.632306
    44   CTAS   209.139999       69.973331
    45    STX   109.290001       69.118534
    46    NOW   938.650024       68.126463
    47    SPG   169.229996       68.110253
    48   ZBRA        373.5       67.940652
    49    TPR    44.810001       67.811928
    

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

      Ticker Stock Price One Year Return  No of shares to buy
    0    VST      124.18      292.337586                 1610
    1   NVDA  132.649994      189.729104                 1507
    2   PLTR   43.130001      142.303387                 4637
    3    CEG  262.279999      133.892372                  762
    4    NRG   88.559998      128.381874                 2258
    

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

      Ticker  Stock Price No of shares to buy  One Year Price Return  \
    0    VST   124.180000                None             292.337586   
    1   NVDA   132.649994                None             189.729104   
    2   PLTR    43.130001                None             142.303387   
    3    CEG   262.279999                None             133.892372   
    4    NRG    88.559998                None             128.381874   
    
       Six Month Price Return  Three Month Price Return  One Month Price Return  \
    0               76.369412                 35.875208               63.611046   
    1               52.428496                  4.129778               22.720947   
    2               92.372885                 56.041975               24.079411   
    3               37.727767                 21.180846               45.509015   
    4               22.352141                 11.821307               13.625859   
    
      HQM Score  
    0      None  
    1      None  
    2      None  
    3      None  
    4      None  
    

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

       Ticker  Stock Price HQM Score No of shares to buy  One Year Price Return  \
    0     VST   124.180000      98.0                None             292.337586   
    2    PLTR    43.130001      97.0                None             142.303387   
    3     CEG   262.279999      83.0                None             133.892372   
    5    FICO  2024.989990      82.0                None             127.744468   
    15   AXON   432.140015      80.5                None             100.724617   
    
        One Year Return Percentile  Six Month Price Return  \
    0                        100.0               76.369412   
    2                         96.0               92.372885   
    3                         94.0               37.727767   
    5                         90.0               71.430629   
    15                        70.0               35.782065   
    
        Six Month Return Percentile  Three Month Price Return  \
    0                          98.0                 35.875208   
    2                         100.0                 56.041975   
    3                          76.0                 21.180846   
    5                          96.0                 29.184314   
    15                         70.0                 48.420116   
    
        Three Month Return Percentile  One Month Price Return  \
    0                            94.0               63.611046   
    2                           100.0               24.079411   
    3                            64.0               45.509015   
    5                            84.0               12.669699   
    15                           98.0               20.256020   
    
        One Month Return Percentile  
    0                         100.0  
    2                          92.0  
    3                          98.0  
    5                          58.0  
    15                         84.0  
    

## 7. Share allocation

The number of shares to buy for each stock is recalculated:


```python
position_size = portfolio_size/len(hqm_df.index)
hqm_df['No of shares to buy'] = (position_size/hqm_df['Stock Price']).apply(math.floor)
```


```python
print(hqm_df)
```

       Ticker  Stock Price HQM Score  No of shares to buy  One Year Price Return  \
    0     VST   124.180000      98.0                 1610             292.337586   
    2    PLTR    43.130001      97.0                 4637             142.303387   
    3     CEG   262.279999      83.0                  762             133.892372   
    5    FICO  2024.989990      82.0                   98             127.744468   
    15   AXON   432.140015      80.5                  462             100.724617   
    9     RCL   193.029999      79.5                 1036             116.591403   
    6     HWM   103.699997      79.0                 1928             123.955314   
    8     KKR   134.949997      77.5                 1482             119.043857   
    14   TRGP   163.399994      73.5                 1223             101.121754   
    1    NVDA   132.649994      72.0                 1507             189.729104   
    7    AVGO   185.949997      71.0                 1075             119.959293   
    12   ANET   406.929993      71.0                  491             108.971397   
    13    IRM   120.489998      68.5                 1659             108.315614   
    17     TT   399.720001      65.0                  500              93.870389   
    22    MMM   135.009995      61.5                 1481              86.231578   
    40    IBM   234.300003      60.5                  853              71.418886   
    19    MHK   159.000000      58.5                 1257              91.497052   
    10     GE   189.279999      57.5                 1056             113.934371   
    32   ERIE   532.599976      56.0                  375              77.324459   
    27     BK    74.019997      55.5                 2701              80.787239   
    24    URI   813.510010      55.0                  245              83.157110   
    4     NRG    88.559998      54.5                 2258             128.381874   
    18   MPWR   936.010010      53.5                  213              92.238296   
    20    PHM   139.389999      51.5                 1434              89.845527   
    11   GDDY   161.210007      51.0                 1240             113.099806   
    38    PWR   307.609985      48.0                  650              73.419606   
    23   META   590.510010      47.5                  338              84.031815   
    26    AXP   271.420013      46.0                  736              81.985022   
    28   LDOS   167.669998      44.0                 1192              79.886842   
    29    SYF    52.000000      42.5                 3846              78.937473   
    34    WAB   184.220001      42.5                 1085              76.504901   
    46    NOW   938.650024      40.0                  213              68.126463   
    35    TDG  1386.959961      40.0                  144              75.274368   
    37    DHI   185.259995      38.0                 1079              74.234197   
    30     RL   195.729996      35.0                 1021              77.596500   
    48   ZBRA   373.500000      34.0                  535              67.940652   
    39    RTX   123.949997      34.0                 1613              73.303954   
    16   NFLX   727.429993      33.5                  274              94.854275   
    31    PGR   250.759995      33.0                  797              77.405703   
    25   DELL   122.339996      32.5                 1634              83.129183   
    44   CTAS   209.139999      32.0                  956              69.973331   
    21   DECK   160.440002      31.5                 1246              89.291129   
    33   FITB    42.630001      29.0                 4691              77.240827   
    43   NTAP   127.269997      27.5                 1571              70.632306   
    36    DVA   159.300003      26.0                 1255              74.517972   
    45    STX   109.290001      24.0                 1829              69.118534   
    41   KLAC   804.630005      23.5                  248              71.208056   
    42    KEY    16.870001      20.5                11855              70.929065   
    47    SPG   169.229996      20.0                 1181              68.110253   
    49    TPR    44.810001      13.0                 4463              67.811928   
    
        One Year Return Percentile  Six Month Price Return  \
    0                        100.0               76.369412   
    2                         96.0               92.372885   
    3                         94.0               37.727767   
    5                         90.0               71.430629   
    15                        70.0               35.782065   
    9                         82.0               47.808349   
    6                         88.0               59.163307   
    8                         84.0               36.743250   
    14                        72.0               44.210081   
    1                         98.0               52.428496   
    7                         86.0               41.529206   
    12                        76.0               41.187284   
    13                        74.0               61.358667   
    17                        66.0               34.575364   
    22                        56.0               47.555352   
    40                        20.0               28.303301   
    19                        62.0               32.865382   
    10                        80.0               21.460298   
    32                        36.0               36.186808   
    27                        46.0               34.920256   
    24                        52.0               19.823200   
    4                         92.0               22.352141   
    18                        64.0               40.845930   
    20                        60.0               27.818796   
    11                        78.0               28.751703   
    38                        24.0               21.042828   
    23                        54.0               13.818015   
    26                        48.0               25.403078   
    28                        44.0               33.519863   
    29                        42.0               26.884825   
    34                        32.0               25.168793   
    46                         8.0               22.030970   
    35                        30.0               20.037708   
    37                        26.0               25.437389   
    30                        40.0               18.626127   
    48                         4.0               24.537360   
    39                        22.0               23.680898   
    16                        68.0               17.596749   
    31                        38.0               21.745079   
    25                        50.0               -0.103937   
    44                        12.0               25.921755   
    21                        58.0               18.825375   
    33                        34.0               24.818823   
    43                        14.0               25.072019   
    36                        28.0               20.626992   
    45                        10.0               25.874310   
    41                        18.0               17.554765   
    42                        16.0               16.725788   
    47                         6.0               19.287957   
    49                         2.0                9.938648   
    
        Six Month Return Percentile  Three Month Price Return  \
    0                          98.0                 35.875208   
    2                         100.0                 56.041975   
    3                          76.0                 21.180846   
    5                          96.0                 29.184314   
    15                         70.0                 48.420116   
    9                          88.0                 18.569138   
    6                          92.0                 30.291591   
    8                          74.0                 24.168774   
    14                         84.0                 22.983857   
    1                          90.0                  4.129778   
    7                          82.0                  9.359353   
    12                         80.0                 13.164993   
    13                         94.0                 28.259069   
    17                         66.0                 16.293704   
    22                         86.0                 31.490659   
    40                         58.0                 32.549633   
    19                         62.0                 31.079964   
    10                         28.0                 17.070375   
    32                         72.0                 47.282207   
    27                         68.0                 21.326063   
    24                         20.0                 22.379193   
    4                          34.0                 11.821307   
    18                         78.0                 12.795789   
    20                         56.0                 22.618880   
    11                         60.0                 13.432319   
    38                         26.0                 16.492338   
    23                          6.0                 15.286394   
    26                         46.0                 13.980836   
    28                         64.0                 12.846267   
    29                         54.0                  6.584923   
    34                         44.0                 13.462564   
    46                         32.0                 25.297008   
    35                         22.0                 16.621971   
    37                         48.0                 23.936021   
    30                         14.0                  6.855806   
    48                         38.0                 13.760962   
    39                         36.0                 22.506172   
    16                         12.0                 11.440826   
    31                         30.0                 18.430073   
    25                          2.0                -12.759853   
    44                         52.0                 17.096939   
    21                         16.0                  8.327328   
    33                         40.0                 12.390288   
    43                         42.0                 -2.775283   
    36                         24.0                 13.582890   
    45                         50.0                  3.689876   
    41                         10.0                 -5.492433   
    42                          8.0                 12.865373   
    47                         18.0                 15.165027   
    49                          4.0                  3.938445   
    
        Three Month Return Percentile  One Month Price Return  \
    0                            94.0               63.611046   
    2                           100.0               24.079411   
    3                            64.0               45.509015   
    5                            84.0               12.669699   
    15                           98.0               20.256020   
    9                            62.0               22.224321   
    6                            86.0               11.325813   
    8                            78.0               14.306285   
    14                           74.0               13.330561   
    1                            12.0               22.720947   
    7                            20.0               25.876517   
    12                           34.0               24.538636   
    13                           82.0                6.193473   
    17                           50.0               15.482621   
    22                           90.0                3.781996   
    40                           92.0               14.114550   
    19                           88.0                6.099025   
    10                           56.0               13.522785   
    32                           96.0                5.350638   
    27                           66.0                9.643007   
    24                           68.0               15.543909   
    4                            24.0               13.625859   
    18                           28.0               10.567019   
    20                           72.0                5.191653   
    11                           36.0                7.717495   
    38                           52.0               23.066378   
    23                           48.0               17.092919   
    26                           44.0               10.883075   
    28                           30.0                9.103281   
    29                           14.0               12.676053   
    34                           38.0               12.342972   
    46                           80.0                9.126315   
    35                           54.0               11.718644   
    37                           76.0               -1.210475   
    30                           16.0               13.810412   
    48                           42.0               11.542480   
    39                           70.0                3.034076   
    16                           22.0                7.988183   
    31                           60.0                0.693848   
    25                            2.0               14.722428   
    44                           58.0                2.466869   
    21                           18.0                8.075587   
    33                           26.0                4.956123   
    43                            6.0               11.169539   
    36                           40.0                3.744708   
    45                            8.0                7.593704   
    41                            4.0               12.778569   
    42                           32.0                6.234261   
    47                           46.0                3.214192   
    49                           10.0                8.841398   
    
        One Month Return Percentile  
    0                         100.0  
    2                          92.0  
    3                          98.0  
    5                          58.0  
    15                         84.0  
    9                          86.0  
    6                          50.0  
    8                          74.0  
    14                         64.0  
    1                          88.0  
    7                          96.0  
    12                         94.0  
    13                         24.0  
    17                         78.0  
    22                         14.0  
    40                         72.0  
    19                         22.0  
    10                         66.0  
    32                         20.0  
    27                         42.0  
    24                         80.0  
    4                          68.0  
    18                         44.0  
    20                         18.0  
    11                         30.0  
    38                         90.0  
    23                         82.0  
    26                         46.0  
    28                         38.0  
    29                         60.0  
    34                         56.0  
    46                         40.0  
    35                         54.0  
    37                          2.0  
    30                         70.0  
    48                         52.0  
    39                          8.0  
    16                         32.0  
    31                          4.0  
    25                         76.0  
    44                          6.0  
    21                         34.0  
    33                         16.0  
    43                         48.0  
    36                         12.0  
    45                         28.0  
    41                         62.0  
    42                         26.0  
    47                         10.0  
    49                         36.0  
    

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

