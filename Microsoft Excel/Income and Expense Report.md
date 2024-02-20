## Overview 
[Link to the file](https://github.com/ashergeo/My-Portfolio/raw/main/assets/Microsoft%20Excel/Abraham%20Properties%202023.xlsx)

### Data Processing

- **Description:** Initial data import and transformation using Excel's Power Query Editor.
- **Techniques Used:** Get Data from Text/CSV, data cleaning with functions like IF, LEFT, RIGHT, MID, SEARCH, ISBLANK, DATE, COUNTIFS.

### Checking Account Sheet

- **Description:** Detailed accounting of monthly transactions with a summary table for quick insights.
- **Tables:** 
  - Monthly summary table (image format).
  - Detailed transaction table with columns for Date, Type, Description, Debit (-), Credit (+), and Balance.
- **Interactive Features:** Slicer for filtering by month.

### Individual Totals

- **Description:** Sheet to provide individual totals based on unique entries in the Description column from the Checking Account sheet.
- **Data Validation:** Dropdown list of unique entries.
- **Formula Used:** `=FILTER(Table_CheckAcct,Table_CheckAcct[Individual Index]=$D$2)`.

### Check Register

- **Description:** Table with details of issued checks.
- **Columns:** Number, Date, Description, Amount.
- **Interactive Features:** Slicer for filtering by payee (e.g., 'MCK Engineering', 'RC Commissioner of Finance', 'US Tax Service').

## Screenshots

![Checking Account](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Microsoft%20Excel/Checking%20Account.png)  

  
![Individual Accounts](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Microsoft%20Excel/Individual%20Accounts.png)


    
![Check Register](https://github.com/ashergeo/My-Portfolio/blob/main/assets/Microsoft%20Excel/Check%20Register.png)

