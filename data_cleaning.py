"""
Created on Wed Jun  2 11:33:35 2021

@author: Johnny Hsieh
"""
import pandas as pd
import missingno as msno

df = pd.read_csv('dataset/full_data.csv')

# Data Info -------------------------------------------------------------------
df.head()

df.shape # Before Cleaning: (1309, 13), After Cleaing: (1309, 14)

df.describe()

df.info()

# Show missing values
msno.matrix(df, figsize=(10, 10))
fig = msno.bar(df, figsize=(10, 10))
# fig.figure.savefig('images/before_cleaning.png', dpi=500)
# fig.figure.savefig('images/after_cleaning.png', dpi=500)
# -----------------------------------------------------------------------------

# Fill the missing values ('Age', 'Fare', 'Cabin', 'Embarked')-----------------
    
# Use median to fill the missing values in 'Age'.
median = df['Age'].median()
df['Age'].fillna(median, inplace=True)

# Use median to fill the missing values in 'Fare'.
median = df['Fare'].median()
df['Fare'].fillna(median, inplace = True)

# Fill the missing values in 'Cabin'.
df['new_cabin'] = df.Cabin

new_cabin = []
for i in df.Cabin:
  if pd.isnull(i):
    new_cabin.append('N') 
  else:
    deck = str(i).split(' ')[0][0]
    new_cabin.append(deck)

df['new_cabin'] = new_cabin
df.head()

df.new_cabin.value_counts().sort_index()

# Count each cabin's mean fare.
a_mean = df[df['new_cabin']=='A']['Fare'].mean()
b_mean = df[df['new_cabin']=='B']['Fare'].mean()
c_mean = df[df['new_cabin']=='C']['Fare'].mean()
d_mean = df[df['new_cabin']=='D']['Fare'].mean()
e_mean = df[df['new_cabin']=='E']['Fare'].mean()
f_mean = df[df['new_cabin']=='F']['Fare'].mean()
g_mean = df[df['new_cabin']=='G']['Fare'].mean()

data = {'Cabin': ['A', 'B', 'C', 'D', 'E', 'F', 'G'], 'Mean': [a_mean, b_mean, c_mean, d_mean, e_mean, f_mean, g_mean]}
cabin_table = pd.DataFrame(data)
cabin_table.sort_values('Mean')
"""
     Cabin   Mean
6     G   14.205000
5     F   18.079367
0     A   41.244314
3     D   53.007339
4     E   54.564634
2     C  107.926598
1     B  122.383078
"""

def reasign_cabin(cabin_fare):
    
    cabin = cabin_fare[0]
    fare = cabin_fare[1]
    
    if cabin == 'N':
        if (fare >= 122):
            return 'B'
        elif ((fare < 122) and (fare >= 107)):
            return 'C'
        elif ((fare < 107) and (fare >= 54)):
            return 'E'
        elif ((fare < 54) and (fare >= 53)):
            return 'D'
        elif ((fare < 53) and (fare >= 41)):
            return 'A'
        elif ((fare < 41) and (fare >= 18)):
            return 'F'
        else:
            return 'G'
    else:
        return cabin
# Use fare to decide which cabin is the record belongs to. 
df['new_cabin'] = df[['new_cabin', 'Fare']].apply(reasign_cabin, axis=1)

# Use appear most frequently place to fill the missing values in 'Embarked'
embarked_max = df['Embarked'].value_counts().index.tolist()[0]
df['Embarked'].fillna(embarked_max, inplace = True)
# -----------------------------------------------------------------------------

# Export data -----------------------------------------------------------------
df.to_csv('dataset/full_data_cleaned.csv', index=False)


