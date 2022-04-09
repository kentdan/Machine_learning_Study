# %% codeblock 
#example
# modules we'll use
import pandas as pd
import numpy as np
# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet
# %% codeblock 
# read in all our data
professors = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/pakistan_intellectual_capital.csv")
# set seed for reproducibility
np.random.seed(493182721)
# %% codeblock 
professors.head()
# %% codeblock 
# get all the unique values in the 'Country' column
countries = professors['Country'].unique()

# sort them alphabetically and then take a closer look
countries.sort()
countries
# %% codeblock 
# convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
professors['Country'] = professors['Country'].str.strip()
# %% codeblock
# get all the unique values in the 'Country' column
countries = professors['Country'].unique()

# sort them alphabetically and then take a closer look
countries.sort()
countries
# %% codeblock
# get the top 10 closest matches to "south korea"
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# take a look at them
matches
# %% codeblock
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    # get a list of unique strings
    strings = df[column].unique()

    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match

    # let us know the function's done
    print("All done!")
# %% codeblock
replace_matches_in_column(df=professors, column='Country', string_to_match="south korea")
# %% codeblock
# get all the unique values in the 'Country' column
countries = professors['Country'].unique()
# sort them alphabetically and then take a closer look
countries.sort()
countries
# %% codeblock
# exercise
# TODO: Your code here
Graduated_from = professors['Graduated from'].unique()
# remove trailing white spaces
professors['Graduated from'] = professors['Graduated from'].str.strip()
# %% codeblock
matches = fuzzywuzzy.process.extract("usa", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
replace_matches_in_column(df=professors, column='Country', string_to_match="usa", min_ratio=70)
# get all the unique values in the 'Country' column
countries = professors['Country'].unique()
# sort them alphabetically and then take a closer look
countries.sort()
countries
