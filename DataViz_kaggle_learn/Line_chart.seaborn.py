# %% codeblock
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
#%% codeblock
# Path of the file to read
spotify_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# %% codeblock
# Print the first 5 rows of the data
spotify_data.head()
# Print the last five rows of the data
spotify_data.tail()
# %% codeblock
# Line chart showing daily global streams of each song
sns.lineplot(data=spotify_data)
# %% codeblock
# Set the width and height of the figure
plt.figure(figsize=(14,6))
# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
# Line chart showing daily global streams of each song
sns.lineplot(data=spotify_data)
# %% codeblock
list(spotify_data.columns)
# %% codeblock
# Set the width and height of the figure
plt.figure(figsize=(14,6))
# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")
# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")
# Add label for horizontal axis
plt.xlabel("Date")
# %% codeblock
# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")
# %% codeblock
#exercise
#import pandas as pd
#pd.plotting.register_matplotlib_converters()
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
#print("Setup Complete")
# %% codeblock
# Path of the file to read
museum_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/museum_visitors.csv"
# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)
# %% codeblock
# Fill in the line below: How many visitors did the Chinese American Museum
# receive in July 2018?
ca_museum_jul18 = 2620
# Fill in the line below: In October 2018, how many more visitors did Avila
# Adobe receive than the Firehouse Museum?
avila_oct18 = 14658
# %% codeblock
## Line chart showing the number of visitors to each museum over time
sns.lineplot(data=museum_data) # Your code here
# Check your answer
plt.show()
# %% codeblock
# Line plot showing the number of visitors to Avila Adobe over time
sns.lineplot(data=museum_data['Avila Adobe'], label="Avila Adobe")
