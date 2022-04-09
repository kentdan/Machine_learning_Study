# %% codeblock 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
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
df.head()
# %% codeblock 
useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if col.startswith("cat")]
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

    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

    model = XGBRegressor(random_state=42, tree_method='hist',predictor="cpu_predictor")
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)

print(np.mean(scores), np.std(scores))
