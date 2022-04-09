# %% codeblock 

# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
​# %% codeblock 
# Load the data, and separate the target
iowa_file_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
​# %% codeblock 
# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()
​# %% codeblock 
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
​​# %% codeblock 
# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
​​# %% codeblock 
# fit rf_model_on_full_data on all data from the training data
from sklearn.ensemble import RandomForestRegressor
# Define the model. Set random_state to 1
rf_model_on_full_data = RandomForestRegressor(random_state=1)
# fit your model
rf_model_on_full_data.fit(train_X,train_y)
# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model_on_full_data.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
​​# %% codeblock
# path to file you will use for predictions
test_data_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/test.csv'
# read test data file using pandas
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)
​​# %% codeblock
# submition
# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
