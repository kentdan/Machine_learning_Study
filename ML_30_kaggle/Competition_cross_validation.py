# %% codeblock 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
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
# Apply ordinal encoder
final_predictions = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    ordinal_encoder = OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])

    model = XGBRegressor(random_state=fold, n_jobs=4)
    model.fit(xtrain, ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_predictions.append(test_preds)
    print(fold, mean_squared_error(yvalid, preds_valid, squared=False))
# %% codeblock 
preds = np.mean(np.column_stack(final_predictions), axis=1)
sample_submission.target = preds
sample_submission.to_csv("submission_Com_ross_val.csv", index=False)
# %% codeblock 
