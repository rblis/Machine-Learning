import numpy as np
from matplotlib import pyplot
from scipy import optimize
import os

def main():
    '''
    data = np.loadtxt(os.path.join('Data Sets', 'ex2data1.txt'), delimiter=',')
    X, y = data[:, 0:2] , data[:,2]
    #plotData(X,y)
    # Test the implementation of sigmoid function here
    # Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    #costFunction(np.array([-24, 0.2, 0.2]),X,y)
    
    


    # set options for optimize.minimize
    options= {'maxiter': 400}

    # see documention for scipy's optimize.minimize  for description about
    # the different parameters
    # The function returns an object `OptimizeResult`
    # We use truncated Newton algorithm for optimization which is 
    # equivalent to MATLAB's fminunc
    # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
    initial_theta = np.zeros(n+1)
    res = optimize.minimize(costFunction,
                            initial_theta,
                            (X, y),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property
    theta = res.x

    # Print theta to screen
    print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
    print('Expected cost (approx): 0.203\n');

    print('theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
    print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')
    #  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 
    prob = sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85,'
        'we predict an admission probability of {:.3f}'.format(prob))
    print('Expected value: 0.775 +/- 0.002\n')

    # Compute accuracy on our training set
    p = predict(theta, X)
    print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
    print('Expected accuracy (approx): 89.00 %')
    '''
    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).
    data = np.loadtxt(os.path.join('Data Sets', 'ex2data2.txt'), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    
    plotData(X, y)
    # Labels and Legend
    pyplot.xlabel('Microchip Test 1')
    pyplot.ylabel('Microchip Test 2')

    # Specified in plot order
    pyplot.legend(['y = 1 = pass', 'y = 0 = fail'], loc='upper right')
    X = mapFeature(X[:, 0], X[:, 1])



def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total 
        number of polynomial features. 
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).
    
    y : array_like
        The data labels. A vector with shape (m, ).
    
    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the regularized cost function. 
    
    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    
    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ===================== YOUR CODE HERE ======================
    sum = 0.0#cost function sum
    reg_cost_sum = 0.0#regularized cost function sum
    #Sigmoid function has to dot product sum of both theta and X vectors
    for i in range(X.shape[1]):#iterate over all IVs
        sumo = 0.0#partial deriv calculation        
        for j in range(m):#iterate over all data points
            sigmo = sigmoid(np.dot(theta,X[j,:]))
            if i == 0:
                sum += -1*y[j]*np.log(sigmo) - (1-y[j])*np.log(1 - sigmo)
            sumo += (sigmo - y[j])*X[j,i]
        if(i != 0):
            reg_cost_sum += theta[i]**2
            grad[i] = sumo/m + lambda_*theta[i]/m
        else:
            grad[i] = sumo/m
    J = sum/m + reg_cost_sum*lambda_/(2*m)
    
    
    # =============================================================
    return J, grad


def mapFeature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5 
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)
    
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).
    
    X : array_like
        The data to use for computing predictions. The rows is the number 
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X. 
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned 
    logistic regression parameters.You should set p to a vector of 0's and 1's    
    """
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    
    p = [1 if sigmoid(np.dot(theta,X[j,:])) >= .5 else 0 for j in range(m)]
    
    # ============================================================
    return p

def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).
    
    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the 
        intercept has already been added to the input.
    
    y : arra_like
        Labels for the input. This is a vector of shape (m, ).
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
        
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    sum = 0.0
    #Sigmoid function has to dot product sum of both theta and X vectors
    for i in range(X.shape[1]):
        sumo = 0.0
        for j in range(m):
            sigmo = sigmoid(np.dot(theta,X[j,:]))
            if i == 0:
                sum += -1*y[j]*np.log(sigmo) - (1-y[j])*np.log(1 - sigmo)
            sumo += (sigmo - y[j])*X[j,i]
        grad[i] = sumo/m   
    J = sum/m        
    # =============================================================
    return J, grad
    
def sigmoid(z):
    """
    Compute sigmoid function given the input z.
    
    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix. 
    
    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.
        
    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)
    
    
    # ====================== YOUR CODE HERE ======================
    if z.shape == ():
        g = 1/(1+np.exp(-1*z))
    elif z.ndim == 1:
        for k in range(g.shape[0]):
            g[k] = 1/(1+np.exp(-1*z[k]))
    else:
        for k in range(g.shape[0]):
            for j in range(g.shape[1]):
                g[k][j] = 1/(1+np.exp(-1*(z[k,j])))
    # =============================================================
    return g

def plotData(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.
    
    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset. 
    
    y : array_like
        Label values for the dataset. A vector of size (M, ).
    
    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.  

    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

    """
    # Create New Figure
    fig = pyplot.figure()

    # ====================== YOUR CODE HERE ======================
    pos = y ==1
    neg = y ==0
    pyplot.plot(X[pos,0], X[pos,1], '^g')
    pyplot.plot(X[neg, 0], X[neg,1], 'vm')
    pyplot.xlabel('Exam 1 score')
    pyplot.ylabel('Exam 2 score')
    pyplot.legend(['Admitted', 'Not admitted'])
    
    # ============================================================


if __name__ == '__main__':
    main()
pyplot.show()