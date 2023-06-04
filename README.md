# python-numpy-pandas
```python

import pandas as pd
import numpy as np

# 1

# A Series is a one-dimensional labeled array that can hold any data type in Python.
# It is similar to a column in a spreadsheet or a SQL table.

# Create a Series from a list
data = [10, 20, 30, 40, 50]
s = pd.Series(data)
print(s)

# 2
# A DataFrame is a two-dimensional labeled data structure with columns of potentially different types.
# It is similar to a table in SQL or a spreadsheet.

# Create a DataFrame from a dictionary
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)
print(df)

# 3
# Python Function
my_list = [1, 2, 3, 4, 5]
print(type(my_list))

# 4
# The tail() function is used to return the last n rows of a DataFrame or Series.
# By default, it returns the last 5 rows.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Get the last 2 rows
print(df.tail(2))

# 5
# The shape attribute returns the dimensions of a DataFrame or Series as a tuple.
# For a DataFrame, it returns the number of rows and columns.
# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Get the shape of the DataFrame
print(df.shape)

# 6
# The size attribute returns the number of elements in a DataFrame or Series.
# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Get the size of the DataFrame
print(df.size)

# 7
# The index() function is used to access the index labels of a DataFrame or Series.
# It returns the index object associated with the data.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Get the index labels of the Series
print(s.index)

#8
# The columns() function is used to access the column labels of a DataFrame.
# It returns the column index object associated with the DataFrame.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Get the column labels of the DataFrame
print(df.columns)

#9
# The head() function is used to return the first n rows of a DataFrame or Series.
# By default, it returns the first 5 rows.
# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Get the first 3 rows of the DataFrame
print(df.head(3))

# 10
# The sort_values() function is used to sort a DataFrame or Series by one or more columns.
# It arranges the data in ascending or descending order based on the specified columns.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Sort the DataFrame by 'Age' column in ascending order
df_sorted = df.sort_values(by='Name')

print(df_sorted)

# 11
# The count() function is used to count the non-null values in a DataFrame or Series.
# It returns the number of non-null elements.

# Create a Series with missing values
data = [10, 20, None, 40, 50]
s = pd.Series(data)

# Count the non-null values in the Series
print(s.count())

# 12
# The value_counts() function is used to count the occurrence of unique values in a DataFrame or Series.
# It returns a Series with the unique values as the index and their respective counts as the values.

# Create a Series
data = [10, 20, 30, 40, 10, 20, 10, 30, 40]
s = pd.Series(data)

# Count the occurrence of unique values in the Series
print(s.value_counts())


# 13
# The rename() function is used to change the name of a DataFrame column or index labels.
# It returns a new DataFrame with the updated names.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Rename the 'Age' column to 'Age Group'
df_renamed = df.rename(columns={'Age': 'Age Group'})

print(df_renamed)

# 14
# The dropna() function is used to remove missing or null values row from a DataFrame.
# It returns a new DataFrame with the missing values dropped.

# Create a DataFrame with missing values

data = {'A': [1, 2, np.nan, 4],
        'B': [np.nan, 6, 7, 8]}
df = pd.DataFrame(data)
print(df)
# Drop the rows with missing values
df_dropped = df.dropna()

print(df_dropped)

# 15
# The min() function is used to find the minimum value in a DataFrame or Series.
# It returns the smallest element.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Find the minimum value in the Series
print(s.min())

#16
# The nsmallest() function is used to find the n smallest elements in a DataFrame or Series.
# It returns a new DataFrame or Series with the n smallest values.

# Create a Series
data = [10, 50, 30, 20, 40]
s = pd.Series(data)

# Get the 3 smallest values in the Series
smallest_values = s.nsmallest(3)

print(smallest_values)


# 17
# The nlargest() function is used to find the n largest elements in a DataFrame or Series.
# It returns a new DataFrame or Series with the n largest values.

# Create a Series
data = [10, 50, 30, 20, 40]
s = pd.Series(data)

# Get the 2 largest values in the Series
largest_values = s.nlargest(2)

print(largest_values)

# 18
# The max() function is used to find the maximum value in a DataFrame or Series.
# It returns the largest element.

# Create a Series
data = [10, 50, 30, 20, 40]
s = pd.Series(data)

# Find the maximum value in the Series
print(s.max())

# 19
# The mean() function is used to calculate the mean (average) value of a DataFrame or Series.
# It returns the arithmetic mean.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Calculate the mean of the Series
print(s.mean())

# 20
# The median() function is used to find the median value of a DataFrame or Series.
# It returns the middle value when the data is sorted.

# Create a Series
data = [10, 20, 300, 400, 500]
s = pd.Series(data)

# Find the median of the Series
print(s.median())

# 21
# The mode() function is used to find the mode (most frequent value) in a DataFrame or Series.
# It returns a new DataFrame or Series with the mode values.

# Create a Series
data = [10, 20, 30, 20, 40, 30, 20]
s = pd.Series(data)

# Get the mode values in the Series
mode_values = s.mode()

print(mode_values)

# 22
# The std() function is used to calculate the standard deviation of a DataFrame or Series.
# It measures the variability or dispersion of the data.
# It returns the standard deviation value.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Calculate the standard deviation of the Series
print(s.std())

# 23
# The agg() function is used to apply aggregation functions to a DataFrame or Series.
# It allows you to compute multiple aggregations simultaneously and returns a new DataFrame or Series with the aggregated values.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Apply aggregation functions to the 'Age' column
aggregated_values = df['Age'].agg(['min', 'max', 'mean'])

print(aggregated_values)

# 24
# The describe() function is used to generate descriptive statistics of a DataFrame or Series.
# It provides information such as count, mean, standard deviation, minimum value, maximum value, and percentiles.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Generate descriptive statistics of the Series
statistics = s.describe()

print(statistics)


# 25
# The insert() function is used to insert a column into a specific location within a DataFrame.
# It allows you to specify the column index and the values for the new column.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data)

# Insert a new column named 'Gender' at index 1
df.insert(1, 'Gender', ['M', 'F', 'M', 'F'])

print(df)










```
