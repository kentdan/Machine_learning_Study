# %% codeblock
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# %% codeblock
# Path of the file to read
iris_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/iris.csv"
# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")
# Print the first 5 rows of the data
iris_data.head()
# %% codeblock
# Histogram
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
# %% codeblock
# KDE plot
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
# %% codeblock
# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
# %% codeblock
#Paths of the files to read
iris_set_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/iris_setosa.csv"
iris_ver_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/iris_versicolor.csv"
iris_vir_filepath = "/Users/danielkent/Documents/Code/Python3/Dataset/Data_visualisation/iris_virginica.csv"
# Read the files into variables
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")
# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()
# %% codeblock
# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)
# Add title
plt.title("Histogram of Petal Lengths, by Species")
# Force legend to appear
plt.legend()
# %% codeblock
# KDE plots for each species
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)
# Add title
plt.title("Distribution of Petal Lengths, by Species")
# %% codeblock
#excercise
# Paths of the files to read
cancer_b_filepath = "../input/cancer_b.csv"
cancer_m_filepath = "../input/cancer_m.csv"
# Fill in the line below to read the (benign) file into a variable cancer_b_data
cancer_b_data = pd.read_csv(cancer_b_filepath,index_col="Id")
# Fill in the line below to read the (malignant) file into a variable cancer_m_data
cancer_m_data =  pd.read_csv(cancer_m_filepath,index_col="Id")
# %% codeblock
# Print the first five rows of the (benign) data
cancer_b_data.head()
# %% codeblock
# Print the first five rows of the (malignant) data
cancer_m_data.head()
# %% codeblock
# Fill in the line below: In the first five rows of the data for benign tumors, what is the
# largest value for 'Perimeter (mean)'?
max_perim = 87.46
# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?
mean_radius = 20.57
# %% codeblock
# Histograms for benign and maligant tumors
sns.distplot(a=cancer_b_data['Area (mean)'], label="Benign", kde=False)
sns.distplot(a=cancer_m_data['Area (mean)'], label="Malignant", kde=False)
plt.legend()
# %% codeblock
# KDE plots for benign and malignant tumors
sns.distplot(a=cancer_b_data['Radius (worst)'], label="Benign", kde=False)
sns.distplot(a=cancer_m_data['Radius (worst)'], label="Malignant", kde=False)
# %% codeblock
