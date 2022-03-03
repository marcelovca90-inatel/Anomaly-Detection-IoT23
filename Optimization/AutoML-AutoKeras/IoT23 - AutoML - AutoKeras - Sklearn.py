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


# In[2]:


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TensorFlow.")

import tensorflow as tf
print("\nTensorFlow version: ", tf.__version__)
print("\nIs GPU available?", tf.test.is_gpu_available())
print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("\nPhysical Devices: ", tf.config.list_physical_devices('GPU'))

from tensorflow.python.client import device_lib
print("\nLocal devices:", device_lib.list_local_devices())


# In[3]:


SEED = 10
limit_rows = None


# In[4]:


print(f"Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[5]:


def load_csv(name):
    if limit_rows is None:
        full_filename = f'../Data Preprocessing/sklearn/full/iot23_combined_{name}.csv'
    else:
        full_filename = f'../Data Preprocessing/sklearn/semi/iot23_combined_{int(limit_rows/1000)}k_{name}.csv'
    
    df = pd.read_table(filepath_or_buffer=full_filename, header=None, sep=',').infer_objects().to_numpy()
    
    return df.ravel() if df.shape[1] == 1 else df


# In[6]:


X_train, X_test, y_train, y_test = load_csv('X_train'), load_csv('X_test'), load_csv('y_train'), load_csv('y_test')

print('X_train',X_train.shape,'\ny_train',y_train.shape)
print('X_test',X_test.shape,'\ny_test',y_test.shape)


# In[7]:


clf = ak.StructuredDataClassifier(overwrite=True, max_trials=10, seed=SEED)


# In[8]:


print(f"Fitting started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

clf.fit(X_train, y_train, validation_split=0.25)

print(f"Fitting finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[9]:


print(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

print(f"Results: [test loss, test acc] = {clf.evaluate(X_test, y_test)}")

print(f"Evaluation finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[10]:


tf.keras.utils.plot_model(clf.export_model())


# In[11]:


print(f"Execution finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

