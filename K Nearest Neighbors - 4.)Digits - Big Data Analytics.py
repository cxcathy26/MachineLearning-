
# coding: utf-8

# In[2]:


from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

scaler = StandardScaler()  

from sklearn.datasets import load_digits
digits = load_digits()
import time #clock
import os
import psutil
time_start = time.clock()

X = digits['data']
y = digits['target']

#SPLIT DATA INTO TRAINING AND TESTING SETS 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  


pred = classifier.predict(X_test)  
print(confusion_matrix(y_test, pred))  
print(classification_report(y_test, pred))  

time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

