# %% codelock
from tensorflow import keras
from tensorflow.keras import layers
#import library
# %% codelock
# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
# %% codelock
# Setup plotting
import matplotlib.pyplot as plt​
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
​# %% codelock
#excercise
import pandas as pd
import tensorflow as tf
# %% codelock
red_wine = pd.read_csv('/Users/danielkent/Documents/Code/Dataset/Deep_learning_kaggle/red-wine.csv')
red_wine.head()
# %% codelock
red_wine.shape
# %% codelock
# the target is quality
input_shape = (11,)
# %% codelock
from tensorflow import keras
from tensorflow.keras import layers
# %% codelock
#define a linear model appropriate for this task
model =keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])
# %% codelock
# YOUR CODE HERE
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))
# %% codelock
import tensorflow as tf
import matplotlib.pyplot as plt
# %% codelock
model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()
# %% codelock
