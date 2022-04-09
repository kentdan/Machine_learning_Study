#advantage of mutual information is that it can detect any kind of relationship,
#while correlation only detects linear relationships.
#easy to use and interpret,
#computationally efficient,
#theoretically well-founded,
#resistant to overfitting, and,
#able to detect any kind of relationship
#The actual usefulness of a feature depends on the model you use it with.
# %% codeblock 
#example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")

df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/autos.csv")
df.head()
# %% codeblock 
X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int
# %% codeblock 
# mutual information regression
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
# %% codeblock 
#plot mi score
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
# %% codeblock
# the high-scoring curb_weight feature exhibits a strong relationship with price, the target.
sns.relplot(x="curb_weight", y="price", data=df);
# %% codeblock
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);
# %% codeblock

###exercise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
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
# %% codeblock
# Load data
df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/ames.csv")
# %% codeblock
# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
# %% codeblock
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
# %% codeblock
features = ["YearBuilt", "MoSold", "ScreenPorch"]
sns.relplot(
    x="value", y="SalePrice", col="variable", data=df.melt(id_vars="SalePrice", value_vars=features), facet_kws=dict(sharex=False),
);
# %% codeblock
X = df.copy()
y = X.pop('SalePrice')
mi_scores = make_mi_scores(X, y)
# %% codeblock
print(mi_scores.head(20))
# print(mi_scores.tail(20))  # uncomment to see bottom 20
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))
# plot_mi_scores(mi_scores.tail(20))  # uncomment to see bottom 20
#Location: Neighborhood
#Size: all of the Area and SF features, and counts like FullBath and GarageCars
#Quality: all of the Qual features
#Year: YearBuilt and YearRemodAdd
#Types: descriptions of features and styles like Foundation and GarageType
# %% codeblock
sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen");
# %% codeblock
feature = "GrLivArea"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);
# %% codeblock
#The trends lines within each category of BldgType are clearly very different,
#indicating an interaction between these features.
#Since knowing BldgType tells us more about how GrLivArea relates to SalePrice,
#we should consider including BldgType in our feature set.
#The trend lines for MoSold, however, are almost all the same.
#This feature hasn't become more informative for knowing BldgType.
