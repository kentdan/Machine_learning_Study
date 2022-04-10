# %% codeblock
# Importing core libraries
import numpy as np
import pandas as pd
# Importing from Scikit-Learn
from sklearn import model_selection
# %% codeblock
# Loading data
df_train = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train.csv")
# %% codeblock
df_train["kfold"] = -1
# %% codeblock
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=df_train)):
    print(f"fold {fold} train indicies: {len(train_indicies)} validation indicies: {len(valid_indicies)}")
    df_train.loc[valid_indicies, "kfold"] = fold
# %% codeblock
df_train.to_csv("train_folds_10.csv", index=False)
