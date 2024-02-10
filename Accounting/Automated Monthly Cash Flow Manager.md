# <u> Automated Monthly Cash Flow Manager </u>

## Overview

I developed a monthly cash flow manager for a local church using Excel. This system automates the tracking of transactions, categorizes them by type and particulars, and generates a summary table for each month. Download it [here](https://github.com/ashergeo/My-Portfolio/files/14229876/Expense.Register.xlsx)

## How It Works

- Users enter the date, type, particulars, and amount in the 'Register' sheet.
- The 'Summary' sheet automatically updates with categorized transactions for each month.

## Code Snippets

```excel
# Populate the transaction type and particulars
=SORT(UNIQUE(FILTER('Register'!$D:$E, ('Register'!$D:$D<>""  ) + ('Register'!$E:$E<>""))))

# Obtain corresponding summary
=SUMIFS('Register'!$F:$F,'Register'!$D:$D,Summary!$B5,'Register'!$E:$E,Summary!$C5,'Register'!$G:$G,Summary!D$2)
```
## Conclusion
This cash register system has streamlined transaction tracking for the local church, providing an efficient and user-friendly solution.

--- 
</br>

#### *Screenshots:*
![Register](https://github.com/ashergeo/My-Portfolio/assets/145012626/8e2f9564-cd35-44a8-bd94-dcf100f1d8ae)
![Summary](https://github.com/ashergeo/My-Portfolio/assets/145012626/9ab1fe0f-55fd-44c0-9759-6fc015299072)

