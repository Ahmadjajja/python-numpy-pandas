# Final Syllabus Practice

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

# 26
# The loc() function is used to access a group of rows and columns by label(s) or a boolean array in a DataFrame.
# It allows for both label-based and boolean-based indexing.

# Create a DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Emily'],
        'Age': [25, 30, 35, 28]}
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])

# Access a specific row using label-based indexing
print(df.loc['B'])

# Access a specific element using label-based indexing
print(df.loc['C', 'Age'])

# 27
# The append() function is used to append rows of one DataFrame to another DataFrame.
# It returns a new DataFrame with the appended rows.

# Create the first DataFrame
data1 = {'Name': ['John', 'Emma'],
         'Age': [25, 30]}
df1 = pd.DataFrame(data1)

# Create the second DataFrame
data2 = {'Name': ['Ryan', 'Emily'],
         'Age': [35, 28]}
df2 = pd.DataFrame(data2)

# Append the second DataFrame to the first DataFrame
df_appended = df1.append(df2)

print(df_appended)


# 28
# The concat() function is used to concatenate two or more DataFrames along a particular axis.
# It allows you to combine multiple DataFrames into a single DataFrame.

# Create the first DataFrame
data1 = {'Name': ['John', 'Emma'],
         'Age': [25, 30]}
df1 = pd.DataFrame(data1)

# Create the second DataFrame
data2 = {'Name': ['Ryan', 'Emily'],
         'Age': [35, 28]}
df2 = pd.DataFrame(data2)

# Concatenate the DataFrames along the row axis (axis=0)
df_concatenated = pd.concat([df1, df2], axis=0)

print(df_concatenated)

# 29
# The merge() function is used to merge two DataFrames based on a common column or index.
# It allows you to perform database-style joins on the DataFrames.

# Create the first DataFrame
data1 = {'ID': [1, 2, 3],
         'Name': ['John', 'Emma', 'Ryan']}
df1 = pd.DataFrame(data1)

# Create the second DataFrame
data2 = {'ID': [2, 3, 4],
         'Age': [25, 30, 35]}
df2 = pd.DataFrame(data2)

# Merge the DataFrames based on the 'ID' column
df_merged = pd.merge(df1, df2, on='ID')

print(df_merged)

# 30
# The join() function is used to join two DataFrames on their index or on a specified column.
# It combines the columns of the two DataFrames into a single DataFrame based on a common index
# or column values.

# Create the first DataFrame
data1 = {'ID': [1, 2, 3],
         'Name': ['John', 'Emma', 'Ryan']}
df1 = pd.DataFrame(data1)

# Create the second DataFrame
data2 = {'ID': [2, 3, 4],
         'Age': [25, 30, 35]}
df2 = pd.DataFrame(data2)

# Join the DataFrames based on the 'ID' column
df_joined = df1.set_index('ID').join(df2.set_index('ID'))

print(df_joined)

# 31
# describe(percentiles=[]): The describe() function provides descriptive statistics of a DataFrame or Series. By default, it includes statistics such as count, mean, standard deviation, minimum value, and quartiles.
# The percentiles parameter allows you to specify additional percentiles to include in the output.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Generate descriptive statistics with additional percentiles
statistics = s.describe(percentiles=[0.1, 0.9])

print(statistics)

# 32
# The isna() function is used to detect missing or null values in a DataFrame or Series.
# It returns a boolean mask where True indicates the presence of missing values.

# Create a Series with missing values
data = [10, None, 30, None, 50]
s = pd.Series(data)

# Check for missing values
missing_values = s.isna()

print(missing_values)

# 33
# notna(): The notna() function is the opposite of isna(). It is used to detect non-missing or non-null values in a DataFrame or Series.
# It returns a boolean mask where True indicates the presence of non-missing values.

# Create a Series with missing values
data = [10, None, 30, None, 50]
s = pd.Series(data)

# Check for non-missing values
non_missing_values = s.notna()

print(non_missing_values)

# 34
# thresh: The thresh parameter is used in conjunction with the dropna() function.
# It specifies the minimum number of non-null values required for a row or column to be kept.
# Rows or columns with fewer non-null values than the specified threshold are dropped.

# Create a DataFrame
data = {'A': [1, None, None, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': [7, None, None, None, None]}
df = pd.DataFrame(data)

# Drop rows with fewer than 3 non-null values
df_dropped = df.dropna(thresh=3)
print(df_dropped)


# 35
# The fillna() function is used to fill missing or null values in a DataFrame or Series with a specified value or using a specified filling method.
# It returns a new DataFrame or Series with the missing values filled.

# Create a Series with missing values
data = [10, None, 30, None, 50]
s = pd.Series(data)

# Fill missing values with a specified value
filled_values = s.fillna(0)

print(filled_values)

# 36
# The replace() function is used to replace specific values in a DataFrame or Series with new values.
# It can be used to replace individual values or multiple values using a dictionary or other mapping.

# Create a Series with values to be replaced
data = ['A', 'B', 'A', 'C', 'D']
s = pd.Series(data)

# Replace 'A' with 'X' and 'B' with 'Y'
replaced_values = s.replace({'A': 'X', 'B': 'Y'})

print(replaced_values)

# 37
# The len() function is a built-in Python function that returns the length of an object.
# In the context of pandas, it can be used to determine the number of rows in a DataFrame
# or the length of a Series.

# Create a Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)

# Get the length of the Series
series_length = len(s)

print(series_length)


# 38
# The apply() function is used to apply a function along either the rows or columns of a DataFrame.
# It allows you to perform custom operations on each row or column of the DataFrame.

# Create a DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Apply a custom function to calculate the sum of each row
row_sum = df.apply(lambda row: row['A'] + row['B'], axis=1)

print(row_sum)

# 39
# applymap(): The applymap() function is used to apply a function to every element of a DataFrame.
# It allows you to perform element-wise operations on the DataFrame.
# Create a DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Apply a custom function to double each element in the DataFrame
doubled_values = df.applymap(lambda x: x * 2)

print(doubled_values)

# 40
# drop(): The drop() function is used to remove rows or columns from a DataFrame based on specified labels or indices.
# It returns a new DataFrame with the specified rows or columns dropped.

# Create a DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]}
df = pd.DataFrame(data)

# Drop multiple columns by label
df_dropped = df.drop(['B'], axis=1)

print(df_dropped)

# 41
# groupby(): The groupby() function in pandas is used to split a DataFrame into groups based on one or more columns.
# It is typically followed by an aggregation or transformation operation to compute summary statistics or apply a function to each group.

# Create a DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Group the DataFrame by the 'Category' column
grouped = df.groupby('Category')

# Compute the mean value for each group
mean_values = grouped.mean()

print(mean_values)

# 42
# ngroups: The ngroups attribute is used to get the number of groups formed
# after performing a groupby() operation on a DataFrame.

# Create a DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'C'],
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Group the DataFrame by the 'Category' column
grouped = df.groupby('Category')

# Get the number of groups
num_groups = grouped.ngroups

print(num_groups)

# 43
# groups: The groups attribute is used to get a dictionary where the keys are the unique group names
# and the values are the corresponding group labels or indices.

# Create a DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Group the DataFrame by the 'Category' column
grouped = df.groupby('Category')

# Get the groups dictionary
groups_dict = grouped.groups

print(groups_dict)

# 44
# first(): The first() function is used to retrieve the first row or value of each group
# after performing a groupby() operation.

# Create a DataFrame
data = {'Category': ['A', 'A', 'B', 'B', 'B'],
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Group the DataFrame by the 'Category' column
grouped = df.groupby('Category')

# Get the first row of each group
first_rows = grouped.first()

print(first_rows)

# 45
# get_group(): The get_group() function is used to retrieve a specific group from a grouped DataFrame or Series based on the group label.
# It returns a DataFrame or Series containing only the rows that belong to the specified group.

# Create a DataFrame
data = {'Category': ['A', 'A', 'B', 'B', 'B'],
        'Value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Group the DataFrame by the 'Category' column
grouped = df.groupby('Category')

# Get the group labeled 'A'
group_A = grouped.get_group('B')

print(group_A)

# 46
# sort(): The sort() function is used to sort a DataFrame or Series by one or more columns.
# It returns a new DataFrame or Series with the rows sorted in ascending or descending order based on the specified columns.

# Create a DataFrame
data = {'Name': ['John', 'Alice', 'Bob', 'Emily'],
        'Age': [25, 30, 22, 28],
        'Salary': [5000, 6000, 4500, 5500]}
df = pd.DataFrame(data)

# Sort the DataFrame by 'Age' in descending order
sorted_df = df.sort_values('Age', ascending=False)

print(sorted_df)

# 47
# plot(): The plot() function is used to create various types of plots, such as line plots, bar plots, scatter plots, etc.
# It is a convenient wrapper around Matplotlib' s plotting functions and allows you to visualize your data directly from a DataFrame or Series.


# Create a DataFrame
data = {'Year': [2010, 2011, 2012, 2013, 2014],
        'Sales': [100, 150, 200, 180, 220]}
df = pd.DataFrame(data)

# Plot a line graph of the sales data
df.plot(x='Year', y='Sales', kind='line')

# Show the plot
plt.show()


# 48
# polyfit(): The polyfit() function is a NumPy function used to fit a polynomial of a specified degree to a set of data points
# using the method of least squares. It returns the coefficients of the polynomial fit.

# Create sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Fit a polynomial of degree 2 to the data
coefficients = np.polyfit(x, y, 2)

# Generate the polynomial function
polynomial = np.poly1d(coefficients)

# Generate x values for plotting
x_plot = np.linspace(0, 6, 100)

# Plot the data points and the fitted polynomial
plt.scatter(x, y, label='Data')
plt.plot(x_plot, polynomial(x_plot), label='Fitted Polynomial')

# Add labels and a legend to the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()

# 49

# plot.scatter(): The plot.scatter() function is used to create a scatter plot of data points in a DataFrame or Series.
# It allows you to visualize the relationship between two variables by plotting their values as points on a two-dimensional graph.

# Create a DataFrame
data = {'X': [1, 2, 3, 4, 5],
        'Y': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Plot a scatter plot of the data points
df.plot.scatter(x='X', y='Y')

# Show the plot
plt.show()

# 50

# plot.bar(): The plot.bar() function is used to create vertical bar plots or column charts to visualize data in a DataFrame or Series.
# It allows you to compare values across different categories or groups.

# Create a DataFrame
data = {'Category': ['A', 'B', 'C', 'D'],
        'Value': [10, 20, 15, 30]}
df = pd.DataFrame(data)

# Plot a bar chart of the values
df.plot.bar(x='Category', y='Value')

# Show the plot
plt.show()

# 51

# plot.hist(): The plot.hist() function is used to create a histogram to visualize the distribution of a dataset.
# It bins the data into intervals and displays the frequency or count of values falling into each bin.

# Create a DataFrame
data = {'Values': np.random.randn(100)}
df = pd.DataFrame(data)

# Plot a histogram of the values
df.plot.hist()

# Show the plot
plt.show()


# 52
# plot.pie(): The plot.pie() function is used to create a pie chart to visualize the
# proportions of different categories or groups in a dataset.

# Create a DataFrame
data = {'Category': ['A', 'B', 'C', 'D'],
        'Value': [10, 20, 15, 30]}
df = pd.DataFrame(data)

# Plot a pie chart of the values
df.plot.pie(y='Value', labels=df['Category'], autopct='%1.1f%%')

# Show the plot
plt.show()

# 53
# The rename() function is used to change the names of one or more columns or index labels in a DataFrame or Series.
# It returns a new DataFrame or Series with the updated names.

# Create a DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Rename column 'A' to 'X' and column 'B' to 'Y'
df = df.rename(columns={'A': 'X', 'B': 'Y'})

print(df)

# 54
# lower(): The lower() function is a string method that converts all characters in a string to lowercase.
# It returns a new string with lowercase characters.

text = "Hello World"
lower_text = text.lower()

print(lower_text)

# 55
# set_index(): The set_index() function is used to set one or more columns as the index of a DataFrame.
# It returns a new DataFrame with the specified column(s) as the index.

# Create a DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Set column 'A' as the index
df = df.set_index('A')

print(df)

# 56
# reset_index(): The reset_index() function is used to reset the index of a DataFrame or Series.
# It removes the current index and replaces it with a default integer index. It returns a new DataFrame or Series with the reset index.

# Create a DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Set column 'A' as the index
df = df.set_index('A')

# Reset the index
df = df.reset_index()

print(df)

# 57
# sort_index(): The sort_index() function is used to sort a DataFrame or Series based on the index.
# It returns a new DataFrame or Series with the rows sorted based on the index values.

# Create a DataFrame
data = {'A': [5, 6, 4],
        'B': [3, 2, 1]}
df = pd.DataFrame(data)

# Sort the DataFrame based on the index
df = df.sort_index()

print(df)

# 58
# array(): The array() function is used to create an array object in NumPy.
# It takes a Python sequence (such as a list or tuple) as an argument and returns a new array.

# Create an array from a list
my_list = [1, 2, 3, 4, 5]
arr = np.array(my_list)

print(arr)

