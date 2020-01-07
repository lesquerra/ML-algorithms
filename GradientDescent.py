import sys
import numpy as np
import pandas as pd

def GDRisk(fX, y, n):
    """ Gradient Descent Risk function
    
    Args:
        fX: Estimated response vector, applying function f(X)
        y: Actual response vector
        n: Number of observations (in X)
        
    Returns: Returns risk value
    
    """
    R = (1/2*n)*np.sum(np.power(fX - y, 2))
    return(R)

def BetaUpdate(X, fX, y, n, alpha):
    """ Beta Update function
    
    Args:
        X: Train data matrix containing explanatory variables
        fX: Estimated response vector, applying function f(X)
        y: Actual response vector
        n: Number of observations (in X)
        alpha: Learning rate parameter
    
    Returns: Returns a vector containing updates to be applied
    
    """
    update = -alpha*(1/n)*np.matmul((fX - y), X)
    return(update)
    
def GradientDescent(X, y, alpha, n_iter):
    """ Gradient Descent algorithm
    
    Args:
        X: Train data matrix containing explanatory variables
        y: Train data vector containing the response variable. Actual response vector
        alpha: Learning rate parameter
        n_iter: Number of iterations
    
    Returns: Returns the Betas estimated using the Gradient Descent algorithm after n_iter iterations
    
    """
    #Initialize betas
    betas = np.zeros((1, X.shape[1]))
    n = X.shape[0]
    
    for i in range(n_iter):
        pred = np.matmul(betas, X.T)
        R = GDRisk(pred, y, n)
        betas += BetaUpdate(X, pred, y, n, alpha)
        
    return(betas)

if __name__ == "__main__":

    input_data = np.genfromtxt(sys.argv[1], delimiter = ",")
    output_file = str(sys.argv[2])

    # Data Preparation - Scaling
    nrow = input_data.shape[0]
    ncol = input_data.shape[1] - 1
    scaled_data = np.empty_like(input_data)

    for col in range(ncol):
        temp_data = input_data[:, col]
        col_mean = np.mean(temp_data)
        col_std = np.std(temp_data)
        #Scale variables
        scaled_data[:, col] = (temp_data - col_mean)/col_std

    scaled_data = np.column_stack([np.array(np.repeat([1], nrow)), scaled_data])
    scaled_data[:,ncol + 1] = input_data[:, ncol]

    # Split response variable from train data
    X = scaled_data[:,0:(ncol + 1)]
    y = scaled_data[:,(ncol + 1)]
    
    # Define Parameters Grid
    choices = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.95] # learning rate choices analysed
    n_iter = [100, 100, 100, 100, 100, 100, 100, 100, 100, 60]
    full_res = []
    
    # Run Gradient Descent
    for i in range(len(choices)):
        print("#####################", choices[i])
        res = GradientDescent(X, y, choices[i], n_iter[i])
        if len(full_res) == 0:
            full_res = np.append(choices[i], res)
        else:
            full_res = np.vstack((full_res, np.append(choices[i], res)))
    
    # Write Gradient Descent output to file
    np.savetxt(output_file, full_res, delimiter=",")
