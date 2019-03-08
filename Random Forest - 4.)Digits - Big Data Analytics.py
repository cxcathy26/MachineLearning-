
# coding: utf-8

# RANDOM FORESTS ON Digits DATASET

# In[10]:


import pandas as pd #pandas used for data manipulation
import scipy as py 
import numpy as np
import time #clock

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
#from sklearn.datasets import load_breast_cancer
#from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
#from pydataset import data
from sklearn.metrics import classification_report,confusion_matrix

import os
import psutil

time_start = time.clock()
# Set random seed
np.random.seed(0) #ensure random numbers are same number

#DIGITS DATASET
digits = load_digits()
dfdigits = pd.DataFrame(np.transpose(digits.data.reshape(-1, len(digits.data))))
dfdigits['target'] = digits.target

#CREATING TRAINING AND TEST DATA 
from sklearn.model_selection import train_test_split
rf = RandomForestClassifier(n_estimators=30)

X = digits['data']
y = digits['target']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

rf.fit(X_train,y_train)

predictions = rf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
accuracy_score(y_test, predictions)

time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid()) 
print(process.memory_info()[0]) #to get memory usage


# In[12]:


digits = load_digits() 
dfdigits = pd.DataFrame(np.transpose(digits.data.reshape(-1, len(digits.data)))) 
dfdigits['target'] = digits.target

