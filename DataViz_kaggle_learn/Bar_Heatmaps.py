# %% codeblock
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
pd.plotting.register_matplotlib_converters()
import seaborn as sns
print("Setup Complete")
# %% codeblock
# Path of the file to read
flight_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/flight_delays.csv"
# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")
# %% codeblock
# Print the data
flight_data.head()
# %% codeblock
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
# %% codeblock
# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
# %% codeblock
# Set the width and height of the figure
plt.figure(figsize=(14,7))
# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)
# Add label for horizontal axis
plt.xlabel("Airline")
# %% codeblock
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)
# %% codeblock
#excersice
# Path of the file to read
ign_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/ign_scores.csv"
# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")
#excersice
# Print the data
ign_data # Your code here
# Fill in the line below: What is the highest average score received by PC games,
# for any genre?
high_score =7.759930
# Fill in the line below: On the Playstation Vita platform, which genre has the
# lowest average score? Please provide the name of the column, and put your answer
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'
#excersice
# Bar chart showing average score for racing games by platform
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Average Score for Racing Games, by Platform")
# Bar chart showing average arrival delay for racing games by platform
sns.barplot(x=ign_data.index, y=ign_data['Racing'])
# Add label for vertical axis
plt.ylabel("Average Score for Racing Games")
# %% codeblock
# Heatmap showing average game score by platform and genre
# Set the width and height of the figure
plt.figure(figsize=(14,7))
# Add title
plt.title("average game score ,by platform and genre")
# Heatmap showing average arrival delay for each game score by platform and genre
sns.heatmap(data=ign_data, annot=True)
# Add label for horizontal axis
plt.xlabel("Genre")
# %% codeblock
