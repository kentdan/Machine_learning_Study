# %% codeblock 
#example
# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
# %% codeblock 
# read in our data
landslides = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/landslide_catalog_2007_2016.csv")

# set seed for reproducibility
np.random.seed(324123)
# %% codeblock 
landslides.head()
# %% codeblock 
# print the first few rows of the date column
print(landslides['date'].head())
# %% codeblock 
# check the data type of our date column
landslides['date'].dtype
# %% codeblock 
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
# %% codeblock 
# print the first few rows
landslides['date_parsed'].head()
# %% codeblock
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# %% codeblock
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()
# plot the day of the month
sns.histplot(day_of_month_landslides, kde=False, bins=31)
# %% codeblock
#exercise
# modules we'll use
#import pandas as pd
#import numpy as np
#import seaborn as sns
#import datetime
# read in our data
earthquakes = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/sig_earthquake_1965_2016.csv")
# %% codeblock
#find Date
earthquakes.head()
# %% codeblock
#date
print(earthquakes['Date'].head())
# %% codeblock
earthquakes[3378:3383]
date_lengths = earthquakes.Date.str.len()
date_lengths.value_counts()
# %% codeblock
indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
earthquakes.loc[indices]
# %% codeblock
earthquakes['Date'].dtype
# %% codeblock 
# create a new column, date_parsed, with the parsed dates
earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")
# %% codeblock 
# print the first few rows
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head()
# %% codeblock 
