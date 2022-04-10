# %% codeblock
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# %% codeblock
# Path of the file to read
insurance_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/insurance.csv"
# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)
# %% codeblock
insurance_data.head()
# %% codeblock
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# %% codeblock
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# %% codeblock
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
# %% codeblock
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
# %% codeblock
sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
# %% codeblock
#exercise
# Path of the file to read
candy_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/candy.csv"
# Fill in the line below to read the file into a variable candy_data
candy_data = pd.read_csv(candy_filepath, index_col= 'id')
# %% codeblock
candy_data.head()
# Fill in the line below: Which candy was more popular with survey respondents:
# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)
more_popular = '3 Musketeers'
# Fill in the line below: Which candy has higher sugar content: 'Air Heads'
# or 'Baby Ruth'? (Please enclose your answer in single quotes.)
more_sugar = 'Air Heads'
# %% codeblock
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])
# %% codeblock
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent']) # Your code here
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'],hue=candy_data['chocolate'])
# %% codeblock
# Color-coded scatter plot w/ regression lines
sns.lmplot(x="sugarpercent", y="winpercent", hue="chocolate", data=candy_data)
# %% codeblock
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'
sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])
