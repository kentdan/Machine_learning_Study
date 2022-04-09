​
# %% codeblock
import pandas as pd
# %% codeblock
# save filepath to variable for easier access
melbourne_file_path = '/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melb_test = pd.read_csv(melbourne_file_path)
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
​# %% codeblock 

# drop
# Fill in the line below: get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
​# %% codeblock 

# imputation
from sklearn.impute import SimpleImputer
# Fill in the lines below: imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

​# %% codeblock 
# Preprocessed training and validation features
# Imputation
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns
​# %% codeblock 

# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))
# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)
​# %% codeblock
# path to file you will use for predictions

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)
​​# %% codeblock
# submition
# Run the code to save predictions in the format used for competition scoring
