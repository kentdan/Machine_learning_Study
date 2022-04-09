# %% codeblock 
#example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

accidents = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/accidents.csv")
autos = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/autos.csv")
concrete = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/concrete.csv")
customer = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/customer.csv")
# %% codeblock 
autos["stroke_ratio"] = autos.stroke / autos.bore
autos[["stroke", "bore", "stroke_ratio"]].head()
# %% codeblock 
#formula of engine displacement (measure power)
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)
# %% codeblock 
# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);
# %% codeblock 
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ["RoadwayFeatures"]].head(10)
# %% codeblock 
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

concrete[components + ["Components"]].head(10)
# %% codeblock 
#breaking down features
#ID numbers: '123-45-6789'
#Phone numbers: '(999) 555-0123'
#Street addresses: '8241 Kaggle Ln., Goose City, NV'
#Internet addresses: 'http://www.kaggle.com
#Product codes: '0 36000 29145 2'
#Dates and times: 'Mon Sep 30 07:06:05 2013'
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

customer[["Policy", "Type", "Level"]].head(10)
# %% codeblock 
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()
# %% codeblock 
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["State", "Income", "AverageIncome"]].head(10)
# %% codeblock 
customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

customer[["State", "StateFreq"]].head(10)
# %% codeblock 
# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)
# %% codeblock 
#exercise
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
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
# Prepare data
df = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/FE_data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")
# %% codeblock 
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = df.GrLivArea/df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF+df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF+df.OpenPorchSF+df.EnclosedPorch+df.Threeseasonporch+ df.ScreenPorch
# %% codeblock 
#interaction
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")
# Multiply
X_2 = X_2.mul(df.GrLivArea, axis=0)
# %% codeblock 
#count features
X_3 = pd.DataFrame()

X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)
# %% codeblock 
df.MSSubClass.unique()
# %% codeblock 
X_4 = pd.DataFrame()
# splitting the first underscore
X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
# %% codeblock 
X_5 = pd.DataFrame()
# %% codeblock 
X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
​# %% codeblock 
#join
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)
