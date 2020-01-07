import sys
import numpy as np
from copy import deepcopy

def Perceptron(X, y):
    """ Perceptron algorithm
    
    Args:
        X: Train data matrix containing explanatory variables
        y: Train data vector containing the response variable 
    
    Returns: Returns Perceptron weights over all iterations run until convergence
    
    """
    nrow = X.shape[0]
    ncol = X.shape[1]
    
    X = np.insert(X, obj = X.shape[1], values = 1, axis = 1)
    weights = np.zeros(X.shape[1])
    n_iter = 0
    full_weights = np.zeros(X.shape[1])
    
    while True:
        w_initial = deepcopy(weights)
        for i in range(nrow):
            fx = np.dot(weights, X[i])
            if fx > 0:
                fx = 1
            else:
                fx = -1	
            
            if y[i]*fx <= 0:
                weights += y[i]*X[i]
                 
        bias = np.linalg.norm(weights - w_initial, ord = 1)
        n_iter += 1 
        
        full_weights = np.vstack((full_weights, weights))
        if bias == 0:
            return(full_weights)
            
if __name__ == "__main__":

    input_data = np.genfromtxt(sys.argv[1], delimiter=',')
    output_file = sys.argv[2]
    
    # Split response variable from train data
    X = data[:, 0:ncol] 
    y = data[:, -1]
    
    # Train Perceptron
    full_weights = Perceptron(X, y)
    
    # Write Perceptron output to file
    np.savetxt(output_file, full_weights, delimiter=",")
    