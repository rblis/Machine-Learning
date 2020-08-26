import numpy
import matplotlib.pyplot
import os
from mpl_toolkits.mplot3d import Axes3D 

def main():
    data = numpy.loadtxt(os.path.join('Data Sets', 'ex1data2.txt'), delimiter=',')
    #Size of house(x0) | number of bedrooms(x1) | price of the house(y) |
    #Size of house is in square feet and has a much larger magnitude than the other data
    #There that collumn of data needs to be normalized
    X = data[:,:2]
    y = data[:,2]
    m = y.size
    print(m)    
    X_norm, mu, sigma = featureNormalize(X)
    X = numpy.concatenate([numpy.ones((m,1)), X_norm], axis=1)
    #print(X.shape[])

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
             
        n = X.shape[1]
        temp = numpy.zeros(n)
        for k in range(n):#data points
            grandsum = 0.0
            for i in range(m):#thetas
                sum = 0.0
                for j in range(n):#IVs
                    sum += X[i,j]*theta[j]
                sum -= y[i]                
                sum *= X[i,k]
                grandsum += sum
            temp[k] = theta[k] - (alpha/m)*grandsum
        theta = temp.copy()           
            
        
        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    grandsum = 0.0
    n = X.shape[1]
    for k in range(m):
        sum = 0.0
        for j in range(n):
            sum += X[k,j]*theta[j]
        sum -= y[k]
        grandsum += sum**2
    J = (1/(2*m))*grandsum
    
    # ==================================================================
    return J


def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'numpy.mean' and 'numpy.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = numpy.zeros(X.shape[1])
    sigma = numpy.zeros(X.shape[1])
    
    # =========================== YOUR CODE HERE =====================
    for k in range(X.shape[1]):
        mu[k] = numpy.mean(X[:,k])
        sigma[k] = numpy.std(X[:,k])
        X_norm[:,k] = [(xi - mu[k])/sigma[k] for xi in X[:,k]]   

    
   
    
    # ================================================================
    return X_norm, mu, sigma












if __name__ == '__main__':
    main()