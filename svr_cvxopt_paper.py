# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:21:28 2020

@author: Arshad
"""


import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
#X,y=make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
from matplotlib import pyplot as plt

from sklearn import preprocessing
X=preprocessing.scale(X)
y=preprocessing.scale(y)


             
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVR(object):

    def __init__(self, kernel=linear_kernel, C=1 ,epsilon=0.5):
        self.kernel = kernel
        self.C = C
        self.epsilon=e
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        e=self.epsilon
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
                
        p1=np.vstack((K, -K))
        p2=np.vstack((-K,K))
        P = cvxopt.matrix(np.hstack((p1,p2)))
        q = cvxopt.matrix(e*np.vstack((np.ones((n_samples,1)),np.ones((n_samples,1))))-np.vstack((y.reshape(-1,1),-y.reshape(-1,1))))
        A = cvxopt.matrix(np.hstack((np.ones((1,n_samples)),-np.ones((1,n_samples)))))
        b = cvxopt.matrix(0.0)
        #print(P.size,q.size,A.size,b.size)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(2*n_samples) * -1))
            h = cvxopt.matrix(np.zeros(2*n_samples))
        else:
            tmp1 = np.diag(np.ones(2*n_samples) * -1)
            tmp2 = np.identity(2*n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(2*n_samples)
            tmp2 = np.ones(2*n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        a1=a[:len(a)//2]
        
        a2=a[len(a)//2:]
        #for i in range(len(a1)):
            #print(a1[i]," ",a2[i])
        a=a1-a2
        #print(a)
        # Support vectors have non zero lagrange multipliers
        sv = abs(a) > 1e-5
        ind = np.arange(len(a))[sv]
        #print(ind)
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        
        ab=0
        cnt=0
        
        self.b = np.zeros(len(self.a))
        for n in range(len(self.a)):
            
            cnt=cnt+1
            self.b[n] += self.sv_y[n]
            self.b[n]-=e
                #print(ind[n])
            #print(K[ind[n],sv])
            for i in range(len(self.a)):
                self.b[n]-=(self.a[n]*K[ind[n],ind[i]])
            #self.b[n] -= np.sum(self.a[n] * K[ind[n],sv])
            ab=ab+self.b[n]
        #print(self.b)   
        self.b = ab/cnt
        #print("fsh,jf",self.b)
        
        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(a)):
                self.w += a[n] * X[n]
            #print(self.w)
        else:
            self.w = None

    def predict(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b


if __name__ == "__main__":
    import pylab as pl


    def split_train(X1, y1,a):
        X_train = X1[a:]
        y_train = y1[a:]
        return X_train, y_train

    def split_test(X1, y1,a):
        X_test = X1[:a]
        y_test = y1[:a]
        
        return X_test, y_test


    def test_linear(X,y,C,e,ratio):
        a=(ratio*len(X))//1
        a=int(round(a))
        X_train, y_train = split_train(X, y,a)
        X_test, y_test = split_test(X, y,a)

        clf = SVR(kernel=linear_kernel, C=C ,epsilon=e)
        clf.fit(X_train, y_train)
        #print(clf.w)
        print(clf.C,clf.epsilon)
        y_predict = clf.predict(X_test)
        sum=0;
        for i in range(len(X_test)):
            sum=sum+(y_test[i]-y_predict[i])**2
            
        print("sum error ",sum/len(X_test))
        #print(y_predict[10],y_test[10])
        plt.scatter(y_test, y_predict)

       
    def test_non_linear(X,y,kernel,C,e,ratio):
        a=(ratio*len(X))//1
        a=int(round(a))
        X_train, y_train = split_train(X, y,a)
        X_test, y_test = split_test(X, y,a)

        clf = SVR(kernel,C,e)
        clf.fit(X_train, y_train)
        
        y_predict = clf.predict(X_test)
        sum=0;
        for i in range(len(X_test)):
            sum=sum+(y_test[i]-y_predict[i])**2
            
        print("sum error ",sum/len(X_test))
        
        plt.scatter(y_test, y_predict)

    def test_soft(X,y,C,e,ratio):
        a=(ratio*len(X))//1
        a=int(round(a))
        X_train, y_train = split_train(X, y,a)
        X_test, y_test = split_test(X, y,a)

        clf = SVR(C=1000.1)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        sum=0;
        for i in range(len(X_test)):
            sum=sum+(y_test[i]-y_predict[i])**2
            
        print("sum error ",sum/len(X_test))
        #print(y_predict[10],y_test[10])
        plt.scatter(y_test, y_predict)
        
    def k_fold(X,y,C=10,e=0.3,n_folds=5):
        clf = SVR(linear_kernel,C,e)
        
        spl=len(y)//n_folds
        
        for i in range(n_folds):
            tr1=np.arange(0,i*spl)
            tr2=np.arange(i*spl,(i+1)*spl)
            tr3=np.arange((i+1)*spl,len(X))
            train=np.hstack((tr1,tr3))
            test=tr2
            
            X_train=X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
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
    
#all of the methods are listed below
    #test_linear(X,y,C,e,test_train_ratio)
    #test_non_linear(X,y,polynomial_kernel,C,e,test_train_ratio)
    #test_non_linear(X,y,gaussian_kernel,C,e,test_train_ratio)
    #test_soft(X,y,C,e,test_train_ratio)
    k_fold(X,y,C,e,n_folds)  