# %% codeblock
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# %% codeblock
# Path of the file to read
spotify_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/spotify.csv"
# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# Line chart
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)
# %% codeblock
# Change the style of the figure to the "dark" theme
sns.set_style("dark")
# Line chart
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)
# %% codeblock
#Seaborn has five different themes: (1)"darkgrid", (2)"whitegrid", (3)"dark", (4)"white", and (5)"ticks"
