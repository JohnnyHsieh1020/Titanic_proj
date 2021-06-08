"""
Created on Tue Apr 13 14:18:47 2021

@author: Johnny Hsieh
"""
import pandas as pd
import numpy as np

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

df_train['train_test'] = 1
df_test['train_test'] = 0

df_test['Survived'] = np.NaN

df = pd.concat([df_train, df_test])

df.columns
size_of_data = {'records': df.shape[0], 'columns': df.shape[1]}
print(size_of_data) # {'records': 1309, 'columns': 13}

df.to_csv('full_data.csv', index=False)
