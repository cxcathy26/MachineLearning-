
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier #MULTILAYER PERCEPTRON CLASSIFIER MODEL
from sklearn.metrics import classification_report,confusion_matrix
import time #clock
cancer = load_breast_cancer()
scaler = StandardScaler()

import os
import psutil
time_start = time.clock()

X = cancer['data'] 
y = cancer['target'] 

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

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30)) #can change more parameters, tried 30 bc number of features is 30
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#can also extract MLP weights and biases if needed
time_elapsed = (time.clock() - time_start) #need all code in one block to measure time elapsed 
print(time_elapsed)
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

# store the predicted probabilities for class 1
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
from sklearn import metrics
print(metrics.roc_auc_score(y_test, y_pred_prob))

