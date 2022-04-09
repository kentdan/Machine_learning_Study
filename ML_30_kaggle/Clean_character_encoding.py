# %% codeblock 
#example
# modules we'll use
import pandas as pd
import numpy as np
# helpful character encoding module
import chardet
# set seed for reproducibility
np.random.seed(41321213)
# %% codeblock 
# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
type(before)
# %% codeblock 
# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")

# check the type
type(after)
# %% codeblock
# take a look at what the bytes look like
after
# %% codeblock
# convert it back to utf-8
print(after.decode("utf-8"))
# %% codeblock
# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been
# replaced with the underlying byte string for the unknown character :(
# %% codeblock
kickstarter_2016 = pd.read_csv("./clean_kickstarter_project/ks-projects-201612.csv")
#UnicodeDecodeError we got when we tried to decode UTF-8 bytes as if they were ASCII!
#This tells us that this file isn't actually UTF-8
# %% codeblock
# look at the first ten thousand bytes to guess the character encoding
with open("./clean_kickstarter_project/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# %% codeblock
# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("./clean_kickstarter_project/ks-projects-201612.csv", encoding='Windows-1252')
# look at the first few lines
kickstarter_2016.head()
# %% codeblock
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# %% codeblock
#exercise
# modules we'll use
#import pandas as pd
#import numpy as np
# helpful character encoding module
#import chardet
# set seed for reproducibility
np.random.seed(45987145)
# %% codeblock
sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))
# %% codeblock
before = sample_entry.decode("big5-tw")
new_entry = before.encode()
# %% codeblock
police_killings = pd.read_csv("/Users/danielkent/Documents/Code/Python3/Dataset/Police_shooting_US/PoliceKillingsUS.csv", encoding='Windows-1252')
# %% codeblock
police_killings.to_csv("my_police_killings.csv")
