# Bank Statement Analysis and Cash Flow Report

## Overview

This project involves processing raw bank statements of a firm to create a user-friendly and interactive Excel file that summarizes the financial transactions. The Excel file utilizes various data processing and visualization techniques to provide insights into the firm’s financial activities.

**Download the Excel file [here](https://github.com/AsherGeorgy/My-Portfolio/raw/refs/heads/main/Microsoft%20Excel/assets/Cash%20Flow%20Report.xlsx).**

## Objectives and Skills

**Objectives:**
- To transform raw bank statement data into a structured, easily navigable Excel file.
- To provide clear and concise summaries of financial transactions for improved decision-making.

**Skills/Tools Demonstrated:**
- **Microsoft Excel**: Data processing, summarization, and visualization.
- **Power Query**: Data import and transformation.
- **Excel Functions**: IF, LEFT, RIGHT, MID, SEARCH, ISBLANK, DATE, COUNTIFS.
- **Data Validation**: Creating interactive dropdowns.
- **Conditional Formatting**: Highlighting selected data.
- **Slicers**: For interactive filtering of data tables.

## Problem Statement and Background

Firms often need to analyze their financial transactions to manage cash flow, monitor expenses, and prepare for audits. This project aims to process raw bank statements into an organized Excel workbook, enhancing readability and enabling interactive exploration of data. This allows for quick insights into financial health and more efficient decision-making processes.

## Approach and Methodology

1. **Data Processing**:
   - Imported raw data from text/CSV files using Excel's Power Query Editor.
   - Applied data cleaning techniques using Excel functions (e.g., IF, LEFT, RIGHT) to transform the data into a usable format.
   - Addressed missing values and standardized the format for consistency.

2. **Checking Account Sheet**:
   - Created a detailed accounting of monthly transactions.
   - Included a summary table for quick insights into monthly performance.
   - Added interactive features like slicers for filtering transactions by month.

3. **Individual Totals Sheet**:
   - Developed a summary sheet to show totals based on unique entries from the Checking Account sheet.
   - Used data validation to create a dropdown list of unique entries for easy selection.
   - Applied conditional formatting to highlight selected entries in the summary table.
   - Employed Excel’s `FILTER` function to dynamically display transactions related to the selected entry.

4. **Check Register Sheet**:
   - Constructed a detailed table for issued checks, including columns for Number, Date, Description, and Amount.
   - Integrated slicers for filtering the check register by payee, enhancing user interaction.

## Results and Insights

- **Checking Account Sheet**: Provides a comprehensive overview of all transactions with the ability to filter by month for detailed analysis.
- **Individual Totals Sheet**: Allows users to quickly view and analyze total amounts for specific transaction descriptions, aiding in the identification of major expense categories or significant financial activities.
- **Check Register Sheet**: Offers a detailed log of issued checks, making it easy to track payments and manage cash flow efficiently.

## Visuals

- **Checking Account**:
  
  ![Checking Account](assets/Checking%20Account.png)
  
- **Individual Accounts**:
  
  ![Individual Accounts](assets/Individual%20Accounts.png)
  
- **Check Register**:
  
  ![Check Register](assets/Check%20Register.png)

## Replication Instructions

1. **Download the Excel File**

2. **Open the File in Microsoft Excel**:
   - Make sure you have Excel 2016 or later, as this version supports all the features used in the workbook.

3. **Explore the Sheets**:
   - Navigate through the different sheets (`Data Processing`, `Checking Account`, `Individual Totals`, and `Check Register`) to explore the processed data and summaries.
   - Utilize the interactive features like slicers and dropdowns to filter and analyze data as needed.

4. **Analyze Your Own Data**:
   - To use this template for your own data, replace the raw data in the `Data Processing` sheet.
   - Follow the same data cleaning steps in Power Query or adjust formulas to suit your specific data format.

---


