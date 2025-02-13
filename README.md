DataFrame Operations in Python

Overview

This document provides an overview of essential DataFrame operations using the Pandas library in Python. Pandas is a powerful data manipulation tool built on top of the Python programming language.

Prerequisites

Ensure you have Pandas installed before running any of the operations below.

pip install pandas

Importing Pandas

import pandas as pd

Creating a DataFrame

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)

Viewing Data

print(df.head())  # First 5 rows
print(df.tail())  # Last 5 rows
print(df.info())  # Summary of the DataFrame
print(df.describe())  # Statistical summary

Selecting Data

print(df['Name'])  # Selecting a single column
print(df[['Name', 'Age']])  # Selecting multiple columns
print(df.iloc[0])  # Selecting a row by index
print(df.loc[0, 'Name'])  # Selecting a specific value

Filtering Data

filtered_df = df[df['Age'] > 28]  # Filtering rows where Age > 28
print(filtered_df)

Adding a New Column

df['Salary'] = [50000, 60000, 70000]  # Adding a new column
print(df)

Modifying Data

df.at[1, 'Age'] = 32  # Modifying a specific value
print(df)

Deleting Columns and Rows

df.drop(columns=['Salary'], inplace=True)  # Removing a column
df.drop(index=1, inplace=True)  # Removing a row
print(df)

Sorting Data

df.sort_values(by='Age', ascending=False, inplace=True)  # Sorting by Age in descending order
print(df)

Handling Missing Values

df.fillna(value={'Age': df['Age'].mean()}, inplace=True)  # Filling missing values
df.dropna(inplace=True)  # Dropping rows with missing values

Grouping and Aggregation

grouped_df = df.groupby('City').mean()  # Grouping by City and calculating the mean
print(grouped_df)

Merging DataFrames

df2 = pd.DataFrame({'Name': ['Alice', 'Charlie'], 'Gender': ['F', 'M']})
merged_df = pd.merge(df, df2, on='Name', how='left')
print(merged_df)

Saving and Loading Data

df.to_csv('data.csv', index=False)  # Saving to a CSV file
df_loaded = pd.read_csv('data.csv')  # Loading from a CSV file
print(df_loaded)

Conclusion

These are some of the fundamental DataFrame operations using Pandas. Mastering these will help you efficiently manipulate and analyze data in Python.

