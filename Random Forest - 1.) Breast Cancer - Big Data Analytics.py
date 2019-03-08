
# coding: utf-8

# Random Forests on Breast Cancer Dataset

# In[1]:


import pandas as pd #pandas used for data manipulation
import scipy as py 
import numpy as np
import time #clock

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

import os
import psutil

cancer = load_breast_cancer()

time_start = time.clock()
# Set random seed
np.random.seed(0) #ensure random numbers are same number


from sklearn.model_selection import train_test_split
rf = RandomForestClassifier()

X = cancer['data']
y = cancer['target']

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
print(process.memory_info().rss)

