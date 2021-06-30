# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:10:45 2020

@author: Arshad
"""


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
#X,y=make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)

from sklearn import preprocessing
X=preprocessing.scale(X)
y=preprocessing.scale(y)

             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class SVM:
    def __init__(self, kernel=linear_kernel, C=10 ,epsilon=0.5):
        self.kernel = kernel
        self.C = C
        self.epsilon=epsilon
        if self.C is not None: self.C = float(self.C)
        
    def fit(self, X, y):
#Initializing values and computing H. Note the 1. to force to float type
        m,n = X.shape

        y = y.reshape(-1,1) * 1.
        X_dash = y * X
        H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format - as previously
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))
#Run solver
        # solve QP problem
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
#==================Computing and printing parameters===============================#
        w = ((y * alphas).T @ X).reshape(-1,1)
        S = (alphas > 1e-4).flatten()
        b = y[S] - np.dot(X[S], w)
        b1=0
        print("\n    %d support vectors out of %d points" % (len(y[S]),len(y)))
        for i in range(len(b)):
            b1=b1+b[i]
        self.w=w
        self.b=b1/len(b)
        #print(self.b,b1/len(b))
        
    def predict(self, X):
        W=self.w
        return -(np.dot(X, W[:-1]) + self.b)/W[-1]
        
        
if __name__ == "__main__":
    import pylab as pl
    
    def gen_data(X,y,e):
        # generate training data in the 2-d case
        y = y.reshape(-1,1) * 1.
        y1=y-e
        y2=y+e
        l=len(y1)

        X1=np.hstack((X, y1))
        X2=np.hstack((X, y2))
        
        y1=np.ones(l)*-1
        y2=np.ones(l)
        return X1, y1, X2, y2
    
    def split_train(X1, y1, X2, y2,a):
        X1_train = X1[a:]
        y1_train = y1[a:]
        X2_train = X2[a:]
        y2_train = y2[a:]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1,e,a):
        X1_test = X1[:a]
        X_test = X1_test[:,:-1]
        y_test = X1_test[:,-1]+e
        return X_test, y_test
    
    def test_linear(X,y,C=10,e=0.8,ratio=0.2):
        a=(ratio*len(X))//1
        a=int(round(a))
        
        clf = SVM(linear_kernel,C,e)
        X1, y1, X2, y2 = gen_data(X,y,e)
        X_train, y_train = split_train(X1, y1, X2, y2,a)
        X_test, y_test = split_test(X1,e,a)

        
        clf.fit(X_train, y_train)
        #print(clf.w)
        w=clf.w

        y_predict = clf.predict(X_test)
        sum=0;
        for i in range(len(X_test)):
            sum=sum+(y_test[i]-y_predict[i])**2
            
        print ("    parameters are C = %f and epsilon = %f "%(C,e))
        print("    mean squared error = ",sum/len(X_test))
        plt.scatter(y_test, y_predict)
        
    def test_soft(X,y,C=1000.1,e=0.5,ratio=0.2):
        a=(ratio*len(X))//1
        a=int(round(a))
        clf = SVM(linear_kernel,C,e)
        X1, y1, X2, y2 = gen_data(X,y,e)
        X_train, y_train = split_train(X1, y1, X2, y2,a)
        X_test, y_test = split_test(X1,e,a)

        clf.fit(X_train, y_train)
        w=clf.w
        #print(clf.b)
        #print(w)
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        sum=0;
        for i in range(len(X_test)):
            sum=sum+(y_test[i]-y_predict[i])**2
            
        print("sum error ",sum/len(X_test))
        plt.scatter(y_test, y_predict)
        
    def merge_train(X1,y1,X2,y2):
        
        X_train = np.vstack((X1, X2))
        y_train = np.hstack((y1, y2))
        return X_train, y_train
    
    def test_data(X1,e):
        X_test=X1[:,:-1]
        y_test=X1[:,-1]+e
        return X_test,y_test
        
    def k_fold(X,y,C=10,e=0.8,n_folds=5):
        clf = SVM(linear_kernel,C,e)
        X1, y1, X2, y2 = gen_data(X,y,e)
        
        spl=len(y)//n_folds
        
        for i in range(n_folds):
            tr1=np.arange(0,i*spl)
            tr2=np.arange(i*spl,(i+1)*spl)
            tr3=np.arange((i+1)*spl,len(X))
            train=np.hstack((tr1,tr3))
            test=tr2
            
            X_train, y_train = merge_train(X1[train], y1[train], X2[train], y2[train])
            X_test, y_test = test_data(X1[test],e)
            clf.fit(X_train, y_train)
        #print(clf.w)
            w=clf.w

            y_predict = clf.predict(X_test)
            sum=0;
            for i in range(len(X_test)):
                sum=sum+(y_test[i]-y_predict[i])**2
            
            print("sum error ",sum/len(X_test))    
    
    
    
    C=1
    e=0.3
    n_folds=5
    test_train_ratio=0.2
    test_linear(X,y,C,e,test_train_ratio)                 # for linear
    #test_soft(X,y,e=e,ratio=test_train_ratio)           #for softlinear
    #k_fold(X,y,C,e,n_folds)                              #for k_fold