# %% codelock
from tensorflow import keras
from tensorflow.keras import layers
# %% codelock
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer
    layers.Dense(units=1),
])
# %% codelock
#excercise
import tensorflow as tf
# Setup plotting
import matplotlib.pyplot as plt
​# %% codelock
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
​# %% codelock
import pandas as pd
​# %% codelock
concrete = pd.read_csv('/Users/danielkent/Documents/Code/Dataset/Deep_learning_kaggle/concrete.csv')
concrete.head()
​# %% codelock
#CompressiveStrength
input_shape = (8,)
from tensorflow import keras
from tensorflow.keras import layers
​# %% codelock
# Now create a model with three hidden layers, each having 512 units and the ReLU activation.
#Be sure to include an output layer of one unit and no activation,
#and also input_shape as an argument to the first layer.
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
​# %% codelock
#rewrite to use activation layers
model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])
​# %% codelock
# YOUR CODE HERE: Change 'relu' to 'elu', 'selu', 'swish'... or something else
activation_layer = layers.Activation('relu')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function
​# %% codelock
plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
​# %% codelock
