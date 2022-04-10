# %% codeblock 
# Importing core libraries
import numpy as np
import pandas as pd
import joblib
# Importing from Scikit-Learn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# %% codeblock
# Loading data
X_train = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/train.csv")
X_test = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/30-days-of-ml/test.csv")
# %% codeblock
# Preparing data as a tabular matrix
y_train = X_train.target
X_train = X_train.set_index('id').drop('target', axis='columns')
X_test = X_test.set_index('id')
# %% codeblock
# Pointing out categorical features
categoricals = [item for item in X_train.columns if 'cat' in item]
# %% codeblock
# Dealing with categorical data using get_dummies
dummies = pd.get_dummies(X_train.append(X_test)[categoricals])
X_train[dummies.columns] = dummies.iloc[:len(X_train), :]
X_test[dummies.columns] = dummies.iloc[len(X_train): , :]
del(dummies)
# %% codeblock
# Dealing with categorical data using OrdinalEncoder (only when there are 3 or more levels)
ordinal_encoder = OrdinalEncoder()
X_train[categoricals[3:]] = ordinal_encoder.fit_transform(X_train[categoricals[3:]]).astype(int)
X_test[categoricals[3:]] = ordinal_encoder.transform(X_test[categoricals[3:]]).astype(int)
X_train = X_train.drop(categoricals[:3], axis="columns")
X_test = X_test.drop(categoricals[:3], axis="columns")
# %% codeblock
# Feature selection (https://www.kaggle.com/lucamassaron/tutorial-feature-selection-with-boruta-shap)
important_features = ['cat1_A', 'cat1_B', 'cat5', 'cat8', 'cat8_C', 'cat8_E', 'cont0',
                      'cont1', 'cont10', 'cont11', 'cont12', 'cont13', 'cont2', 'cont3',
                      'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9']

categoricals = ['cat5', 'cat8']

X_train = X_train[important_features]
X_test = X_test[important_features]
# %% codeblock
# Stratifying the data
pca = PCA(n_components=16, random_state=0)
km = KMeans(n_clusters=32, random_state=0)
pca.fit(X_train)
km.fit(pca.transform(X_train))
print(np.unique(km.labels_, return_counts=True))
y_stratified = km.labels_
# %% codeblock 
# Creating your folds for repeated use (for instance, stacking)
folds = 10
seed = 42
# %% codeblock
X_train['kfold'] = -1
# %% codeblock
# Checking the produced folds
skf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
for k, (train_idx, validation_idx) in enumerate(skf.split(X_train, y_stratified)):
    print(f"fold {k} train idxs: {len(train_idx)} validation idxs: {len(validation_idx)}")
    X_train.loc[validation_idx,"kfold"] = k
# %% codeblock
X_train.to_csv("train1_folds.csv",index=False)
# %% codeblock
fold_idxs = list(skf.split(X_train, y_stratified))
