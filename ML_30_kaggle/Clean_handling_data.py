# %% codeblock 
#example
# modules we'll use
import pandas as pd
import numpy as np
# %% codeblock
# read in all our data
nfl_data = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/NFL_clean/NFL Play by Play 2009-2017 (v4).csv")
# %% codeblock
# look at the first five rows of the nfl_data file.
# I can see a handful of missing data already!
nfl_data.head()
# %% codeblock
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()
# %% codeblock
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# %% codeblock
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
# %% codeblock
# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)
# %% codeblock
# remove all the rows that contain a missing value
nfl_data.dropna()
# %% codeblock
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# %% codeblock
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# %% codeblock
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
# %% codeblock
# replace all NA's with 0
subset_nfl_data.fillna(0)
# %% codeblock
# replace all NA's the value that comes directly after it in the same column,
# then replace all the remaining na's with 0
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)
# %% codeblock
#exercise
# modules we'll use
import pandas as pd
import numpy as np
# %% codeblock
# read in all our data
sf_permits = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/SF_building_permit/Building_Permits.csv",low_memory=False)
sf_permits.head()
#missing_values_count
missing_values_count = sf_permits.isnull().sum()
missing_values_count[0:10]
total_cells = np.product(sf_permits.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells) * 100
# %% codeblock
#drop rows
sf_permits.dropna()
# %% codeblock
#how many coloum drop
sf_permits_with_na_dropped =sf_permits.dropna(axis=1)
sf_permits_with_na_dropped.head()

# calculate number of dropped column
cols_in_original_dataset = sf_permits.shape[1]
cols_in_na_dropped = sf_permits_with_na_dropped.shape[1]
dropped_columns = cols_in_original_dataset - cols_in_na_dropped
#fill remaining nan to 0
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0).fillna(0)
# %% codeblock
