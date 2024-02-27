# <u> Automated Monthly Cash Flow Manager </u>

## Overview

I developed a monthly cash flow manager for a church using Excel. This system automates the tracking of transactions, categorizes them by type and particulars, and generates a summary table for each month. Download it [here](https://github.com/ashergeo/My-Portfolio/raw/main/assets/Microsoft%20Excel/Expense.Register.xlsx)

## How It Works

- Users enter the date, type, particulars, and amount in the 'Register' sheet.
- The 'Summary' sheet automatically updates with categorized transactions for each month.

## Formulas Used

#### Populate the transaction type and particulars:  
    = SORT(UNIQUE(FILTER('Register'!$D:$E, ('Register'!$D:$D<>""  ) + ('Register'!$E:$E<>""))))

#### Obtain corresponding summary:  
    = SUMIFS('Register'!$F:$F,'Register'!$D:$D,Summary!$B5,'Register'!$E:$E,Summary!$C5,'Register'!$G:$G,Summary!D$2)

## Conclusion
This cash register system has streamlined transaction tracking for the church, providing an efficient and user-friendly solution.

--- 
</br>

#### Screenshots:
![Register](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Microsoft%20Excel/Register.png)

---

![Summary](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Microsoft%20Excel/Summary.png)

---


