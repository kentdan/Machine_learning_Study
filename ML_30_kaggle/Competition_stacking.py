# %% codeblock 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
# %% codeblock 
df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train_folds.csv")
df_test = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv")

df1 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train_pred_1.csv")
df1.columns = ["id", "pred_1"]
df2 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train_pred_2.csv")
df2.columns = ["id", "pred_2"]
df3 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train_pred_3.csv")
df3.columns = ["id", "pred_3"]
df4 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train_pred_4.csv")
df4.columns = ["id", "pred_4"]
df5 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/train_pred_5.csv")
df5.columns = ["id", "pred_5"]

df_test1 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/test_pred_1.csv")
df_test1.columns = ["id", "pred_1"]
df_test2 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/test_pred_2.csv")
df_test2.columns = ["id", "pred_2"]
df_test3 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/test_pred_3.csv")
df_test3.columns = ["id", "pred_3"]
df_test4 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/test_pred_4.csv")
df_test4.columns = ["id", "pred_4"]
df_test5 = pd.read_csv("/Users/danielkent/Documents/Code/Python3/ML_30_kaggle/test_pred_5.csv")
df_test5.columns = ["id", "pred_5"]

df = df.merge(df1, on="id", how="left")
df = df.merge(df2, on="id", how="left")
df = df.merge(df3, on="id", how="left")
df = df.merge(df4, on="id", how="left")
df = df.merge(df5, on="id", how="left")

df_test = df_test.merge(df_test1, on="id", how="left")
df_test = df_test.merge(df_test2, on="id", how="left")
df_test = df_test.merge(df_test3, on="id", how="left")
df_test = df_test.merge(df_test4, on="id", how="left")
df_test = df_test.merge(df_test5, on="id", how="left")
# %% codeblock 
sample_submission = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv")
useful_features = ["pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]
df_test = df_test[useful_features]
# %% codeblock 
final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    valid_ids = xvalid.id.values.tolist()

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    params = {
        'random_state': 1,
        'booster': 'gbtree',
        'n_estimators': 7000,
        'learning_rate': 0.03,
        'max_depth': 2
    }

    model = XGBRegressor(
        n_jobs=4,
        **params
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
# %% codeblock 
print(np.mean(scores), np.std(scores))
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_1"]
final_valid_predictions.to_csv("level1_train_pred_1.csv", index=False)
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_1"]
sample_submission.to_csv("level1_test_pred_1.csv", index=False)
# %% codeblock 
sample_submission = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv')
useful_features = ["pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]
df_test = df_test[useful_features]
# %% codeblock 
final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    valid_ids = xvalid.id.values.tolist()

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_depth=3)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
# %% codeblock 
print(np.mean(scores), np.std(scores))
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_2"]
final_valid_predictions.to_csv("level1_train_pred_2.csv", index=False)
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_2"]
sample_submission.to_csv("level1_test_pred_2.csv", index=False)
# %% codeblock 
sample_submission = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv')
useful_features = ["pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]
df_test = df_test[useful_features]
# %% codeblock 
final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    valid_ids = xvalid.id.values.tolist()

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    model = GradientBoostingRegressor(n_estimators=500, max_depth=3)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)
# %% codeblock 
print(np.mean(scores), np.std(scores))
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_3"]
final_valid_predictions.to_csv("level1_train_pred_3.csv", index=False)
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_3"]
sample_submission.to_csv("level1_test_pred_3.csv", index=False)
# %% codeblock 
df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train_folds.csv")
df_test = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv")
sample_submission = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv")
# %% codeblock 
df1 = pd.read_csv("level1_train_pred_1.csv")
df2 = pd.read_csv("level1_train_pred_2.csv")
df3 = pd.read_csv("level1_train_pred_3.csv")

df_test1 = pd.read_csv("level1_test_pred_1.csv")
df_test2 = pd.read_csv("level1_test_pred_2.csv")
df_test3 = pd.read_csv("level1_test_pred_3.csv")

df = df.merge(df1, on="id", how="left")
df = df.merge(df2, on="id", how="left")
df = df.merge(df3, on="id", how="left")

df_test = df_test.merge(df_test1, on="id", how="left")
df_test = df_test.merge(df_test2, on="id", how="left")
df_test = df_test.merge(df_test3, on="id", how="left")
# %% codeblock 
df.head()
# %% codeblock 
useful_features = ["pred_1", "pred_2", "pred_3"]
df_test = df_test[useful_features]

final_predictions = []
scores = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)

print(np.mean(scores), np.std(scores))
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_predictions), axis=1)
sample_submission.to_csv("submission_stacking.csv", index=False)
