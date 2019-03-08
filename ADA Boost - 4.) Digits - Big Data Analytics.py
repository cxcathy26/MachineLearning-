
# coding: utf-8

# In[1]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_digits

digits = load_digits()
#scaler = StandardScaler()

dt = DecisionTreeClassifier() 

import time #clock
import os
import psutil
time_start = time.clock()

X = digits['data']
y = digits['target']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create adaboost-decision tree classifer object
clf = AdaBoostClassifier(n_estimators=85,
                         learning_rate=1,
                         random_state=0,base_estimator=dt)

# Train model
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from sklearn import metrics
print(metrics.accuracy_score(y_test, predictions))

