# %% codeblock 
import pandas as pd
# %% codeblock
# save filepath to variable for easier access
melbourne_file_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
melbourne_data.columns
# %% codeblock
#Prediction target and melbourne_features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()
# %% codeblock
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)
# %% codeblock
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
# %% codeblock
# Code you have previously used to load data
# Path of the file to read
iowa_file_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train.csv'
home_data = pd.read_csv(iowa_file_path)
​# %% codeblock
# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)
​# %% codeblock
y = home_data.SalePrice
​# %% codeblock
#step 2
# Create the list of features below
feature_names = ['LotArea' ,'YearBuilt' , '1stFlrSF' , '2ndFlrSF', 'FullBath','BedroomAbvGr','TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = home_data[feature_names]
​# %% codeblock
# Review data
# print description or statistics from X
#print(_)
print(X.describe())
# print the top few lines
#print(_)
print(X.head())
​# %% codeblock
# from _ import _
from sklearn.tree import DecisionTreeRegressor
#specify the model.
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)
# %% codeblock
# Fit the model
iowa_model.fit(X,y)
predictions = iowa_model.predict(X)
print(predictions)
