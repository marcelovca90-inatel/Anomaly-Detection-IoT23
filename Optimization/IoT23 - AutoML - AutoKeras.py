#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
import time
from datetime import datetime
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearnex import patch_sklearn
patch_sklearn()


# In[2]:


SEED = 10


# In[3]:


print(f"Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[4]:


filepath = "../Data Preprocessing/iot23_combined_1000k.csv"
df = pd.read_csv(filepath)


# In[5]:


df


# In[6]:


del df['Unnamed: 0']


# In[7]:


df['label'].value_counts()


# In[8]:


good = 'Benign'
bad = 'PartOfAHorizontalPortScan'
filtered_labels = df['label'].value_counts().index.drop([good,bad])
for label in filtered_labels:
    df.drop(df[df.label == label].index, inplace=True)


# In[9]:


df['label'].value_counts()


# In[10]:


df.loc[(df.label == good), 'label'] = 0
df.loc[(df.label == bad), 'label'] = 1
df = df.astype({'label': int})


# In[11]:


df['label'].value_counts()


# In[12]:


df


# In[13]:


X = df.loc[:, df.columns != 'label']
print(X.shape)


# In[14]:


Y = df['label'].values
print(Y.shape)


# In[15]:


from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
X = MinMaxScaler(feature_range=(0,1)).fit_transform(X,Y)
X = SelectPercentile(percentile=1).fit_transform(X,Y)
print(X.shape)


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED, test_size=0.2)


# In[17]:


model = ak.StructuredDataClassifier(overwrite=True, max_trials=1)


# In[18]:


start = time.time()
print('program start...')
print()

history = model.fit(X_train, Y_train, validation_split=0.25)
print(history)

print()
end = time.time()
print('program end...')
print()
print('time cost: ')
print(end - start, 'seconds')


# In[19]:


best_model = model.export_model()


# In[20]:


print("Evaluate on test data")
results = best_model.evaluate(X_test, Y_test)
print("test loss, test acc:", results)


# In[21]:


print(f"Execution finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

