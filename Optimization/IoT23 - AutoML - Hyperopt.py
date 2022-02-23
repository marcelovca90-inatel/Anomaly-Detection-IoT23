#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearnex import patch_sklearn
patch_sklearn()
from hpsklearn import *
from hyperopt import hp, tpe


# In[ ]:


SEED = 10


# In[ ]:


print(f"Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[ ]:


filepath = "../Data Preprocessing/iot23_combined_1000k.csv"
df = pd.read_csv(filepath)


# In[ ]:


df


# In[ ]:


del df['Unnamed: 0']


# In[ ]:


df['label'].value_counts()


# In[ ]:


good = 'Benign'
bad = 'PartOfAHorizontalPortScan'
filtered_labels = df['label'].value_counts().index.drop([good,bad])
for label in filtered_labels:
    df.drop(df[df.label == label].index, inplace=True)


# In[ ]:


df['label'].value_counts()


# In[ ]:


df.loc[(df.label == good), 'label'] = 0
df.loc[(df.label == bad), 'label'] = 1
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


from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
X = MinMaxScaler(feature_range=(0,1)).fit_transform(X,Y)
X = SelectPercentile(percentile=1).fit_transform(X,Y)
print(X.shape)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED, test_size=0.2)


# In[ ]:


best_results = {}

classifiers = [
    ada_boost('ab'),
    decision_tree('dt'),
    extra_trees('et'),
    gaussian_nb('gnb'),
    gradient_boosting('gb'),
    knn('knn'),
    liblinear_svc('svc-ll'),
    linear_discriminant_analysis('lda',n_components=1),
    multinomial_nb('mnb'),
    one_vs_one('ovo'),
    one_vs_rest('ovr'),
    output_code('oc'),
    passive_aggressive('pa'),
    quadratic_discriminant_analysis('qda'),
    random_forest('rf'),
    sgd('sgd'),
    svc('svc'),
    svc_linear('svc-l'),
    svc_poly('svc-p'),
    svc_rbf('svc-r'),
    svc_sigmoid('svc-s'),
    xgboost_classification('xgb')
]

for clf in classifiers:
    
    clf_name = clf.name.replace('sklearn_','').replace('Classifier','')
    if clf_name == 'switch':
        clf_name = 'SVC-LL'
    elif 'SVC' in clf_name:
        for arg in clf.named_args:
            if arg[0] == 'kernel':
                clf_name += f'-{arg[1].obj.capitalize()[0]}'
    clf_name = ''.join(c for c in clf_name if (c.isupper() or c == '-'))
    
    print(f'\n******************** {clf_name} ********************')
    
    best_results[clf_name] = (0.0, None)
        
    try:
        estim = HyperoptEstimator(classifier=clf,
                                  preprocessing=[],
                                  algo=tpe.suggest,
                                  max_evals=10,
                                  trial_timeout=600,
                                  seed=np.random.default_rng(SEED),
                                  fit_increment=1,
                                  fit_increment_dump_filename=f'hyperopt_increments/{clf_name}.inc',
                                  n_jobs=-1)
    except Exception as e:
        print(f"********** Could not create {clf_name}. Reason: '{str(e)}'. **********")
        
    try:
        estim.fit(X_train, Y_train, random_state=SEED)
    except Exception as e:
        print(f"********** Could not fit {clf_name}. Reason: '{str(e)}'. **********")
        
    try:
        score = estim.score(X_test, Y_test)
        best_model = estim.best_model()
        best_results[clf_name] = (score, best_model)
        print(best_results[clf_name])
    except Exception as e:
        print(f"********** Could not evaluate {clf_name}. Reason: '{str(e)}'. **********")


# In[ ]:


best_results = dict(sorted(best_results.items()))
print(json.dumps(best_results, indent=4, default=str))


# In[ ]:


names = list(best_results.keys())
print(names)
values = list(x[0] for x in best_results.values())
print(values)

plt.figure(figsize=(16,9))
idx = 0
for i in range(len(best_results.keys())):
    plt.bar(names[i],values[i])
    plt.text(idx-0.1,values[i]+0.01,f'{100*values[i]:.1f}%')
    idx += 1
plt.xticks(rotation=45, ha='right')
plt.xticks(range(0,len(best_results)),names)
plt.yticks(np.linspace(0,1,11))
plt.ylim(0,1)
plt.show()


# In[ ]:


print(f"Execution finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

