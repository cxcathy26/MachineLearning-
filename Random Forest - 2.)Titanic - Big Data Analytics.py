
# coding: utf-8

# RANDOM FORESTS ON TITANIC DATASET

# In[17]:


import pandas as pd #pandas used for data manipulation
import scipy as py 
import numpy as np
import time #clock

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
#from sklearn.datasets import load_breast_cancer
#from sklearn.datasets import load_wine
#from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from pydataset import data
from sklearn.metrics import classification_report,confusion_matrix

import os
import psutil

time_start = time.clock()
# Set random seed
np.random.seed(0) #ensure random numbers are same number

#TITANIC DATASET
#data('titanic', show_doc=True) #can predict if they survived can do males and females separetely, also diabetes
titanictemp = data('titanic')


y = pd.factorize(titanictemp['survived'])[0]
clas = pd.factorize(titanictemp['class'])[0]
age = pd.factorize(titanictemp['age'])[0]
sex = pd.factorize(titanictemp['sex'])[0]

titanic = pd.DataFrame(age) #First column is age category 
titanic['class'] = clas
titanic['sex'] = sex
titanic['survived'] = y # 0 = survived, 1 = died

#CREATING TRAINING AND TEST DATA 
from sklearn.model_selection import train_test_split
rf = RandomForestClassifier(n_estimators = 14) #increased number of estimators 

X = titanic.iloc[:,0:3] 
y = titanic['survived']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

rf.fit(X_train,y_train)

predictions = rf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid())
print(process.memory_info()[0])

