# %% codeblock 
#example
import pandas as pd
# %% codeblock 
autos = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/autos.csv")
# %% codeblock 
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
autos[["make", "price", "make_encoded"]].head(10)
# %% codeblock
# smoothing: encoding = weight * in_category + (1 - weight) * overall
#where weight is a value between 0 and 1 calculated from the category frequency.
#An easy way to determine the value for weight is m-estimate
#m-estimate: weight = n / (n + m)
#n is the total number of times that category occurs in the data. The parameter m determines "smoothing factor".
#Larger values of m put more weight on the overall estimate.
#f you chose m=2.0, then the chevrolet category would be encoded with
#60% of the average Chevrolet price plus 40% of the overall average price.
chevrolet = 0.6 * 6000.00 + 0.4 * 13285.03
# %% codeblock 
#example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
# %% codeblock 
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
warnings.filterwarnings('ignore')
# %% codeblock 
df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/movielens1m.csv")
df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))
# %% codeblock 
X = df.copy()
y = X.pop('Rating')
X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]
# %% codeblock 
from category_encoders import MEstimateEncoder
# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)
# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)
# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)
# %% codeblock 
plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating']);
# %% codeblock 
#exercise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
# %% codeblock 
# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
warnings.filterwarnings('ignore')
# %% codeblock 
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
# %% codeblock 
df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/ames.csv")
# %% codeblock 
df.select_dtypes(["object"]).nunique()
# %% codeblock 
df["SaleType"].value_counts()
# %% codeblock 
#The Neighborhood feature promising. It has the most categories of any feature and several categories are rare.
#Others that could be worth considering are SaleType, MSSubClass, Exterior1st, Exterior2nd.
#almost any of the nominal features would be worth trying because of the prevalence of rare categories
# Encoding split
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")
# Training split
X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")
# %% codeblock 
# YOUR CODE HERE: Create the MEstimateEncoder
# Choose a set of features to encode and a value for m
encoder = MEstimateEncoder(
    cols=["Neighborhood"],
    m=1.0,
)
# Fit the encoder on the encoding split
encoder.fit(X_encode, y_encode)
# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)
# %% codeblock 
#compare encoded to targeted
feature = encoder.cols

plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(X_train[feature], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice");
# %% codeblock 
# the score of the encoded set compared to the original set
X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")
# %% codeblock 
# Try experimenting with the smoothing parameter m
# Try 0, 1, 5, 50
m = 0
X = df.copy()
y = X.pop('SalePrice')
# Create an uninformative feature
X["Count"] = range(len(X))
X["Count"][1] = 0  # actually need one duplicate value to circumvent error-checking in MEstimateEncoder
# fit and transform on the same dataset
encoder = MEstimateEncoder(cols="Count", m=m)
X = encoder.fit_transform(X, y)
# Results
score =  score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")
# %% codeblock
plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X["Count"], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice");
# %% codeblock
#how XGBoost was able to get an almost a perfect fit after mean-encoding the count feature?
#Since Count never has any duplicate values, the mean-encoded Count is essentially an exact copy of the target.
#In other words, mean-encoding turned a completely meaningless feature into a perfect feature
#the only reason this worked is because we trained XGBoost on the same set we used to train the encoder.
#If we had used a hold-out set instead, none of this "fake" encoding transferred to the training data.
