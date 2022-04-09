# %% codeblock 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import optuna
# %% codeblock 
df = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train_folds.csv')
df_test = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv')
sample_submission = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv')
# %% codeblock 
useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
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

    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

    model = XGBRegressor(
        random_state=fold,
        tree_method='hist',
        predictor="cpu_predictor"
    )
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
final_valid_predictions.columns = ["id", "pred_1"]
final_valid_predictions.to_csv("train_pred_1.csv", index=False)
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_1"]
sample_submission.to_csv("test_pred_1.csv", index=False)
# %% codeblock 
df = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train_folds.csv')
df_test = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv')
sample_submission = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv')
# %% codeblock 
useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
numerical_cols = [col for col in useful_features if col.startswith("cont")]
df_test = df_test[useful_features]
# %% codeblock 
poly = preprocessing.PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
train_poly = poly.fit_transform(df[numerical_cols])
test_poly = poly.fit_transform(df_test[numerical_cols])
# %% codeblock 
df_poly = pd.DataFrame(train_poly, columns=[f"poly_{i}" for i in range(train_poly.shape[1])])
df_test_poly = pd.DataFrame(test_poly, columns=[f"poly_{i}" for i in range(test_poly.shape[1])])
# %% codeblock 
df = pd.concat([df, df_poly], axis=1)
df_test = pd.concat([df_test, df_test_poly], axis=1)
# %% codeblock 
useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
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

    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

    model = XGBRegressor(
        random_state=fold,
        tree_method='hist',
        predictor="cpu_predictor",
        max_depth=2
    )
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
final_valid_predictions.to_csv("train_pred_2.csv", index=False)
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_2"]
sample_submission.to_csv("test_pred_2.csv", index=False)
# %% codeblock 
df = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train_folds.csv')
df_test = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv')
sample_submission = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/sample_submission.csv')
# %% codeblock 
useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
df_test = df_test[useful_features]
# %% codeblock 
for col in object_cols:
    temp_df = []
    temp_test_feat = None
    for fold in range(5):
        xtrain =  df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        feat = xtrain.groupby(col)["target"].agg("mean")
        feat = feat.to_dict()
        xvalid.loc[:, f"tar_enc_{col}"] = xvalid[col].map(feat)
        temp_df.append(xvalid)
        if temp_test_feat is None:
            temp_test_feat = df_test[col].map(feat)
        else:
            temp_test_feat += df_test[col].map(feat)

    temp_test_feat /= 5
    df_test.loc[:, f"tar_enc_{col}"] = temp_test_feat
    df = pd.concat(temp_df)
# %% codeblock 
useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if col.startswith("cat")]
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

    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

    model = XGBRegressor(
        random_state=fold,
        tree_method='hist',

        predictor="cpu_predictor",
        max_depth=3
    )
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
final_valid_predictions.to_csv("train_pred_3.csv", index=False)
# %% codeblock 
sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_3"]
sample_submission.to_csv("test_pred_3.csv", index=False)
# %% codeblock 
#
# %% codeblock 
df1 = pd.read_csv("train_pred_1.csv")
df2 = pd.read_csv("train_pred_2.csv")
df3 = pd.read_csv("train_pred_3.csv")
# %% codeblock 
df_test1 = pd.read_csv("test_pred_1.csv")
df_test2 = pd.read_csv("test_pred_2.csv")
df_test3 = pd.read_csv("test_pred_3.csv")
# %% codeblock 
df = df.merge(df1, on="id", how="left")
df = df.merge(df2, on="id", how="left")
df = df.merge(df3, on="id", how="left")
# %% codeblock 
df_test = df_test.merge(df_test1, on="id", how="left")
df_test = df_test.merge(df_test2, on="id", how="left")
df_test = df_test.merge(df_test3, on="id", how="left")
# %% codeblock 
df.head()
# %% codeblock 
useful_features = ["pred_1", "pred_2", "pred_3"]
df_test = df_test[useful_features]
# %% codeblock 
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
sample_submission.to_csv("submission.csv", index=False)
