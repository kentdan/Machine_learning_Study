#a large effect for a few predictions, but no effect in general, or
#a medium effect for all predictions.
#SHAP summary plots give us a birds-eye view of feature importance and what is driving it.
# We'll walk through an example plot for the soccer data:
# %% codeblock
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# %% codeblock
data = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/explainability/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
# %% codeblock
import shap  # package used to calculate Shap values
# %% codeblock
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)
# %% codeblock
# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], val_X)
# %% codeblock
#Calculating SHAP values can be slow. It isn't a problem here, because this dataset is small.
#But you'll want to be careful when running these to plot with reasonably sized datasets.
#The exception is when using an xgboost model,
#which SHAP has some optimizations for and which is thus much faster.
# %% codeblock
#SHAP Dependence Contribution Plots
import shap  # package used to calculate Shap values
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)
# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")
# %% codeblock
#excersice
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
​# %% codeblock
data = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/explainability/hospital.csv')
y = data.readmitted
base_features = ['number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures',
                 'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency',
                 'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414',
                 'diabetesMed_Yes', 'A1Cresult_None']
# %% codeblock
# Some versions of shap package error when mixing bools and numerics
X = data[base_features].astype(float)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# For speed, we will calculate shap values on smaller subset of the validation data
small_val_X = val_X.iloc[:150]
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
# %% codeblock
data.describe()
# %% codeblock
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)
shap.summary_plot(shap_values[1], small_val_X)
# %% codeblock
#diag_1_428 vs payer_code_?
# the range of diag_1_428 is wider, largely due to the few points on the far right.
feature_with_bigger_range_of_effects = 'diag_1_428'
# %% codeblock
 #if all dots on the graph are widely spread from each other,
 #that is a reasonable indication that permutation importance is high.
 #Because the range of effects is so sensitive to outliers,
 #permutation importance is a better measure of what's generally important to the model.
# %% codeblock
shap.summary_plot(shap_values[1], small_val_X)
# %% codeblock
shap.dependence_plot('num_lab_procedures', shap_values[1], small_val_X)
# %% codeblock
shap.dependence_plot('num_medications', shap_values[1], small_val_X)
