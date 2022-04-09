# %% codeblock 
import pandas as pd
import numpy as np
from sklearn import model_selection
# %% codeblock 
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
# %% codeblock 
# Read the data
train_data = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train.csv')
test_data = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv')
# %% codeblock 
#train_data.head()
#train_data.target.hist()
# %% codeblock
train_data['kfold'] = -1
# %% codeblock
#Split
kf = model_selection.KFold(n_splits=5,shuffle=True,random_state=42)
 for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=train_data)):
     train_data.loc[valid_indicies,"kfold"] = fold
# %% codeblock
train_data.to_csv("train_folds.csv",index=False)
