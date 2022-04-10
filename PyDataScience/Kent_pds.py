#load dataset
# %% codeblock
import pandas as pd
df =  pd.read_csv('Pokemon.csv')
print(df.head(5))
#read header
df.columns
print(df['Name'][0:5])
print(df[['Name','Type 1','HP']])
# %% codeblock
#read each row
print(df.iloc[1:4] )
#specific
print(df.iloc[2,1] )
df.loc[df['Type 1'] == "Fire"]
#%% codeblock
#describe like summary in R
df.describe()
#%% codeblock
#sorting abc
df.sort_values('Name', ascending = False)
#sorting number
df.sort_values(['Type 1','HP'], ascending = False)
#%% codeblock
#make total
df['Total'] = df.iloc[:,4:10].sum(axis=1)
df.head(5)
#%% codeblock
# make csv file
df.to_csv('modified.csv',index=False)
#%% codeblock
df.loc[(df['HP'] > 70) & (df['Attack'] > 70) ]
#greater than
#string Mega
df.loc[(df['Name'].str.contains('Mega'))]
#groupby
df.groupby(['Type 1']).mean()
#large amount of dataset
# , chunkby and concat
