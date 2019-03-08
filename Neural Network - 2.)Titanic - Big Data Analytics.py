
# coding: utf-8

# In[1]:


#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier #MULTILAYER PERCEPTRON CLASSIFIER MODEL
from sklearn.metrics import classification_report,confusion_matrix
from pydataset import data
import pandas as pd #pandas used for data manipulation
import time #clock
import os
import psutil

scaler = StandardScaler()
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
X_train, X_test, y_train, y_test = train_test_split(X, y)

#neural network may have difficulty converging before max number of iterations is allowed
    #if data is not normalized
#multi-layer perceptron is sensitive to feature scaling so its highly recommended to 
    #scale the data 
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10))
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#can also extract MLP weights and biases if needed

time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
from sklearn import metrics
print(metrics.roc_auc_score(y_test, y_pred_prob))

