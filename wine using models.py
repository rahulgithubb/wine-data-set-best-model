#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
lw=load_wine()


# In[2]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


model_params = {
    'svm' : {
        'model' : svm.SVC(gamma='auto'),
        'params' : {
            'C' : [1,10,20],
            'kernel' : ['rbf','linear','poly']
        }
    },
    'random_forest' : {
        'model' : RandomForestClassifier(),
        'params' : {
            'n_estimators' : [1,5,10]
        }
    },
    'logistic_regression' : {
        'model' :  LogisticRegression(solver='liblinear',multi_class='auto'),
        'params' : {
            'C' : [1,5,10]
        }
    },
    'naive_bayes_gaussian' : {
         'model' : GaussianNB(),
        'params' : {}
        },
    'naive_bayes_multinomial' : {
         'model' : MultinomialNB(),
        'params' : {}
    },
    'Decision_tree' : {
        'model' : DecisionTreeClassifier(),
        'params' : {
            'criterion' : ['gini','entropy'],
        }
    },
    'Knn' : {
        'model' : KNeighborsClassifier(),
        'params' : {
            'n_neighbors' : [3],
             'algorithm' : ['ball_tree'],
              'metric'   : ['minkowski']
           
    }
    }
    }
    


# In[4]:


from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
scores = []

for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(lw.data,lw.target)
    scores.append(
    {
        'model' : model_name,
        'best_score' : clf.best_score_,
        'best_params' : clf.best_params_
    })

df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:




