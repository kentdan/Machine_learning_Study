# %% codeblock 
import pandas as pd
# %% codeblock
# save filepath to variable for easier access
melbourne_file_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
melbourne_data.describe()
