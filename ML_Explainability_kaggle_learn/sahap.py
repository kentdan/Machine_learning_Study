#SHAP values interpret the impact of having a certain value for a given feature in comparison
#to the prediction we'd make if that feature took some baseline value
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
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
my_model.predict_proba(data_for_prediction_array)
# %% codeblock
import shap  # package used to calculate Shap values
# %% codeblock
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
# %% codeblock
shap_display = shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib=True)
# %% codeblock
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap_display = shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib=True)
# %% codeblock
####excersice
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# %% codeblock
data = pd.read_csv('/Users/danielkent/Documents/Code/Python3/Dataset/explainability/hospital.csv')
data.columns
# %% codeblock
y = data.readmitted
base_features = [c for c in data.columns if c != "readmitted"]
X = data[base_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
# %% codeblock
# Use permutation importance as a succinct model summary
# A measure of model performance on validation data would be useful here too
import eli5
from eli5.sklearn import PermutationImportance
# %% codeblock
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
# %% codeblock
# PDP for number_inpatient feature
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
# %% codeblock
feature_name = 'number_inpatient'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)
# %% codeblock
# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
# %% codeblock
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_name = 'time_in_hospital'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)

# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
# %% codeblock
# A simple pandas groupby showing the average readmission rate for each time_in_hospital.
# Do concat to keep validation data separate, rather than using all original data
all_train = pd.concat([train_X, train_y], axis=1)
all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
# %% codeblock
import shap  # package used to calculate Shap values
sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function
def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data, matplotlib=True)
    return shap_display
# %% codeblock
patient_risk_factors(my_model, data)
