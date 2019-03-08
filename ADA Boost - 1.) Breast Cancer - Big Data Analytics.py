
# coding: utf-8

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

import os
import psutil
import time 
time_start = time.clock()
cancer = load_breast_cancer()
#scaler = StandardScaler()

dt = DecisionTreeClassifier() 

X = cancer['data']
y = cancer['target']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create adaboost-decision tree classifer object
clf = AdaBoostClassifier(n_estimators=85,
                         learning_rate=1.2,
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

