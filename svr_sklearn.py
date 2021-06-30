# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:36:27 2020

@author: Arshad
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


X, y = load_boston(return_X_y=True)
#X,y=make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
from matplotlib import pyplot as plt

from sklearn import preprocessing
X=preprocessing.scale(X)
y=preprocessing.scale(y)

def Kfold(C,e):
    
    #scores = []
    clf = SVR(C=C, epsilon=e, gamma = sigma, kernel = 'linear')
    cv = KFold(n_splits=5, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_predict=clf.predict(X_test)
        #scores.append(clf.score(X_test, y_test))
        sum=0;
        for i in range(len(X_test)):
            sum=sum+(y_test[i]-y_predict[i])**2
               
        print("sum error ",sum/len(X_test))
   
    #print(np.mean(scores))

def split_train(X1, y1,a):
        X_train = X1[a:]
        y_train = y1[a:]
        return X_train, y_train
def split_test(X1, y1,a):
        X_test = X1[:a]
        y_test = y1[:a]
        
        return X_test, y_test
    
def train_test_split(C,e,ratio):
    a=(ratio*len(X))//1
    a=int(round(a))
    clf = SVR(C=C, epsilon=e, gamma = sigma, kernel = 'linear')
    X_train, y_train = split_train(X, y,a)
    X_test, y_test = split_test(X, y,a)
    clf.fit(X_train, y_train)
    y_predict=clf.predict(X_test)
    sum=0;
    for i in range(len(X_test)):
        sum=sum+(y_test[i]-y_predict[i])**2
    mse=mean_squared_error(y_test, y_predict)             
    
    print("  ")
    print ("    parameters are C = %f and epsilon = %f "%(C,e))
    print("    mean squared error = ",sum/len(X_test),mse)
    plt.scatter(y_test, y_predict)
    
C=1
e=0.3
#Kfold(C,e)                                #for kfold
test_train_ratio=0.2    
train_test_split(C,e,test_train_ratio)     #for test-train-split