# %% codeblock
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
# %% codeblock
# Load the large model to get the vectors
nlp = en_core_web_lg.load()
review_data = pd.read_csv('/Users/danielkent/Documents/Code/Dataset/nlp_course/yelp_ratings.csv')
review_data.head()
# %% codeblock
