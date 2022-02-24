#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
import time
from datetime import datetime
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


SEED = 10


# In[ ]:


print(f"Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[ ]:


filepath = "../Data Preprocessing/iot23_combined_3200k_4_classes.csv"
df = pd.read_csv(filepath)


# In[ ]:


df


# In[ ]:


del df['Unnamed: 0']


# In[ ]:


df['label'].value_counts()


# In[ ]:


relevant_labels = ['Benign','DDoS','PartOfAHorizontalPortScan','Okiru']

# leave this block uncommented to remove 'irrelevant' labels
if len(df['label'].value_counts()) != len(relevant_labels):
    print(f'DF labels do not match the desired ones; some rows will be dropped.')
    filtered_labels = df['label'].value_counts().index.drop(relevant_labels)
    for label in filtered_labels:
        df.drop(df[df.label == label].index, inplace=True)
    print(df['label'].value_counts())
else:
    print(f'DF labels match the desired ones; no rows will be dropped.')


# In[ ]:


# convert 'label' from categoric to numeric type
for i in range(len(relevant_labels)):
    df.loc[(df.label == relevant_labels[i]), 'label'] = i
df = df.astype({'label': int})


# In[ ]:


df['label'].value_counts()


# In[ ]:


df


# In[ ]:


X = df.loc[:, df.columns != 'label']
print(X.shape)


# In[ ]:


Y = df['label'].values
print(Y.shape)


# In[ ]:


from sklearn.feature_selection import VarianceThreshold, SelectFpr, SelectFdr, SelectFwe, SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

X = VarianceThreshold(threshold=0).fit_transform(X,Y)    # remove constant features (i.e. with zero variance)
print(f'Input shape after variance thresold filter: {X.shape}')

X = MinMaxScaler(feature_range=(0,1)).fit_transform(X,Y) # scale data to [0,1] range
print(f'Input shape after min-max feature scaling:  {X.shape}')

X = VarianceThreshold(threshold=0).fit_transform(X,Y)    # remove constant features (i.e. with zero variance)
print(f'Input shape after variance thresold filter: {X.shape}')

X = SelectPercentile(percentile=1).fit_transform(X,Y)    # select top 1% features according to ANOVA F-value
print(f'Input shape after percentile selection:     {X.shape}')

X = VarianceThreshold(threshold=0).fit_transform(X,Y)    # remove constant features (i.e. with zero variance)
print(f'Input shape after variance thresold filter: {X.shape}')


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED, test_size=0.2)


# In[ ]:


model = ak.StructuredDataClassifier(overwrite=True, max_trials=10)


# In[ ]:


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


# In[ ]:


best_model = model.export_model()
print(best_model)


# In[ ]:


print("Evaluate on test data")
results = best_model.evaluate(X_test, Y_test)
print("test loss, test acc:", results)


# In[ ]:


print(f"Execution finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

