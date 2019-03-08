
# coding: utf-8

# In[7]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.datasets import load_wine
wine = load_wine()
#scaler = StandardScaler()

dt = DecisionTreeClassifier() 
import time #clock
import os
import psutil
time_start = time.clock()

X = wine['data']
y = wine['target']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create adaboost-decision tree classifer object
clf = AdaBoostClassifier(n_estimators=50,
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

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
from sklearn.metrics import recall_score
print(metrics.recall_score(y_test, predictions, average='micro'))

