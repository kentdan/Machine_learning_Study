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
# exercise 4 Model Validation
# %% codeblock 
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# %% codeblock
#split test
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
# %% codeblock
#exercise
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# %% codeblock
# Import the train_test_split function and uncomment
# from _ import _
from sklearn.model_selection import train_test_split
# fill in and uncomment
# train_X, val_X, train_y, val_y = ____
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# %% codeblock
# Specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(train_X, train_y)
# Fit iowa_model with the training data.
# %% codeblock
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
print(val_mae)
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_predictions, val_y)
print(val_mae)
# %% codeblock
