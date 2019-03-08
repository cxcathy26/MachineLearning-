
# coding: utf-8

# In[6]:


from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

scaler = StandardScaler()  

from pydataset import data
import pandas as pd #pandas used for data manipulation
import time #clock
import os
import psutil
time_start = time.clock()

#TITANIC DATASET
#data('titanic', show_doc=True) #can predict if they survived can do males and females separetely
titanictemp = data('titanic')
y = pd.factorize(titanictemp['survived'])[0]
clas = pd.factorize(titanictemp['class'])[0]
age = pd.factorize(titanictemp['age'])[0]
sex = pd.factorize(titanictemp['sex'])[0]

titanic = pd.DataFrame() 
titanic['age'] =age
titanic['class'] = clas
titanic['sex'] = sex
titanic['survived'] = y # 0 = survived, 1 = died

X = titanic.iloc[:,0:3] 
y = titanic['survived']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  

scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

classifier = KNeighborsClassifier(n_neighbors=3)  #changed nearest neighbor from 5 to 3
classifier.fit(X_train, y_train)  


pred = classifier.predict(X_test)  
print(confusion_matrix(y_test, pred))  
print(classification_report(y_test, pred))  

time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid())
print(process.memory_info()[0])

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
from sklearn import metrics
print(metrics.roc_auc_score(y_test, y_pred_prob))

