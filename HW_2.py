#!/usr/bin/env python
# coding: utf-8

# In[39]:


# Anthony Perales
# 801150315
#======================================= Homework 2

# ========================================= Problem 1 ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('diabetes.csv', header = 0)

#num_vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

X = df.drop(['Outcome'],axis=1)
Y = df.pop('Outcome')

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, random_state = 42)



# In[40]:


sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

model = LogisticRegression(solver='liblinear')
model.fit(X_train_std, Y_train)
predicted = model.predict(X_test_std)
report = classification_report(Y_test, predicted)
print(report)


# In[41]:


N = Normalizer()

X_train_N = N.fit_transform(X_train)
X_test_N = N.transform(X_test)

model = LogisticRegression(solver='liblinear')
model.fit(X_train_N, Y_train)
predicted = model.predict(X_test_N)
report = classification_report(Y_test, predicted)
print(report)


# In[42]:


matrix = confusion_matrix(Y_test, predicted)
print(matrix)


# In[43]:


#=========================== Problem 2 ==================================
K_fold = KFold(n_splits = 5, random_state = 42, shuffle=True)
results = cross_val_score(model, X, Y, cv = K_fold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[44]:


K_fold = KFold(n_splits = 10, random_state = 42, shuffle=True)
results = cross_val_score(model, X, Y, cv = K_fold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[115]:


#================================ Problem 3 =======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
cancer = load_breast_cancer()
cancer_data = cancer.data
#num_vars = ['data', 'target_names', 'target']

X = cancer_data[:,0]
X = X.reshape(-1,1)
Y = cancer.pop('target')


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, random_state = 42)


# In[116]:


sc =StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

model = LogisticRegression(solver='liblinear')
model.fit(X_train_std, Y_train)
predicted = model.predict(X_test_std)
report = classification_report(Y_test, predicted)
print(report)


# In[117]:


matrix = confusion_matrix(Y_test, predicted)
print(matrix)


# In[138]:


C = [0.1]

for c in C:
    clfd = LogisticRegression(penalty = 'l1', C=c, solver='liblinear')
    clfd.fit(X_train_std, Y_train)
    print('C:', c)
    print('Training accuracy:', clfd.score(X_train_std, Y_train))
    print('Test accuracy:', clfd.score(X_test_std,Y_test))
    print('')


# In[147]:


#============================ Problem 4 =========================
K_fold = KFold(n_splits = 5, random_state = 42, shuffle=True)
results = cross_val_score(model, X, Y, cv = K_fold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[140]:


K_fold = KFold(n_splits = 10, random_state = 42, shuffle=True)
results = cross_val_score(model, X, Y, cv = K_fold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[141]:


C = [0.1]

for c in C:
    clfd = LogisticRegression(penalty = 'l1', C=c, solver='liblinear')
    clfd.fit(X_train_std, Y_train)
    print('C:', c)
    print('Training accuracy:', clfd.score(X_train_std, Y_train))
    print('Test accuracy:', clfd.score(X_test_std, Y_test))
    print('')


# In[ ]:




