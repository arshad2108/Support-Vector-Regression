# Support-Vector-Regression
The datasets used in this is the Boston housing data.
After having the data I did pre-processing by the sklearn scaling pre-processing function 

## SVR_class(rh-svr)
Methods to call the program:

* test_linear(X,y,C=1,e=0.3,ratio=0.2)
* test_soft(X,y,C=1000.1,e=0.5,ratio=0.2)
* k_fold(X,y,C=1,e=0.3,n_folds=5):
* SVM initialisation:- __init__(self, kernel=linear_kernel, C=1 ,epsilon=0.3):

### PRE-REQ:-
I have implemented this by using cvxopt solver. 
This method have implemented on the basis of the idea you have given in the class by considering y-e and y+e new features and with values - 1 and +1 respectively and evaluating y with $ y=-(w*x+b)/(\mu) $. 


![svr_class](/image/Picture1.jpg "It follows this mathematical equation")

I have implemented this method only for linear types because for nonlinear kernels evaluating y was difficult by above formula of y.
### Algorithm:-
Basically, in this method I have just implemented SVM with additional features y-e and y+e and you and using the line separating both features as the required regressor.
For training and validation I have implemented two types of comparison algorithm namely training test validation split and k fold cross validation method.
In this algorithm, soft margin classifier as well as hard margin classifier can be achieved.
The error function which I have used in this is also mean squared error.


## SVR_sklearn
### Methods to call:
*	train_test_split(C,e,ratio)
*	Kfold(C,e,n_folds)

For the comparing purpose I have used the sklearn SVR with kernel rbf.
For cross validation and training I have used the Kfold cross validation method also it is taken from the sklearn inbuilt Kfold validation split function.
For error measurement I have used mean-squared-error.



## SVR_paper
### Methods to call the program:-
*	test_linear(X,y,C=1,e=0.3,ratio=0.2)
*	test_non_linear(X,y,kernel,C,e,ratio)
*	test_soft(X,y,e,ratio)
*	k_fold(X,y,C=1,e=0.3,n_folds=5)
*	SVR initialisation:- __init__(self, kernel=linear_kernel, C=1 ,epsilon=0.3)

### PRE_REQ:-
I have tried to implement this <a href="https://doi.org/10.1016/S0925-2312(03)00380-1" target="_blank">paper</a> using cvxopt solver.
This method which I have implemented is directly based on solving the minimising problem which we got from epsilon tube assumption.


![svr paper](/image/Picture2.jpg "It follows this mathematical equation")



This method is quite general can be utilised for multiple kernels like linear kernel, polynomial kernel as well as Gaussian kernel.
### Algorithm:-
To formulate this and to overcome ai and ai* I used a single 2n length array of which first n are ai and rest are ai*,  so therefore the equivalent p matrix became [K(x,y) -K(x,y),-K(x,y) K(x,y)]. Similarly I made the rest matrices accordingly and later just formulated what was given in the paper, you given.
In this algorithm, soft margin classifier as well as hard margin classifier can be achieved.
For checking and validation the methods implemented to be used are training test validation split as well as k fold cross validation split.
Error function is used is mean squared error. 


