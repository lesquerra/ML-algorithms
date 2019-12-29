import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import sys

def EuclideanDistance(x, c):
    """ Euclidean Distance calculation
    
    Args:
        x: Data point vector
        c: Cluster centroids vector
        
    Returns: Returns the euclidean distance between vectors x and c
    
    """
    dist = np.sum((x-c)**2,axis=0)
    return dist

def KMeans(data, K, n_iter):
    """ K-Means algorithm
    
    Args:
        data: Numerical non-labelled data matrix
        K: Number of clusters
        n_iter: Number of iterations to be run
        
    Returns: No specific return value. Writes estimated cluster centroids to a file at each iteration.
    
    """
    #Initialize parameters
    n_row, n_col = data.shape
    rand = np.random.choice(n_row - 1, K)
    centerslist = data[rand]
    clusters = [None] * n_row
    
    for i in range(n_iter):
        
        #Step 1 - assign clusters
        for r in range(n_row):
            x = data[r]
            temp_dist = [None] * K
            for k in range(K):
                temp_dist[k] = EuclideanDistance(x, centerslist[k])
            clusters[r] = np.argmin(temp_dist)
        
        #Step 2 - update clusters
        for k in range(K):
            indices = [ind for ind, c in enumerate(clusters) if c == k]
            x = data[indices]
            centerslist[k,] = np.mean(x, axis = 0)
        
        #Write iteration file
        filename = "centroids-" + str(i+1) + ".csv"
        np.savetxt(filename, centerslist, delimiter=",")

  
def EMGMM(data, K, n_iter):
    """ Expectation-Maximization algorithm for Gaussian Mixture Models
    
    Args:
        data: Non-labelled data matrix
        K: Number of clusters
        n_iter: Number of iterations to be run
        
    Returns: No specific return value. Writes estimated cluster distribution parameters to a file at each iteration.
    
    """
    #Initialize parameters
    n_row, n_col = data.shape
    rand = np.random.choice(n_row - 1, K)
    mu = data[rand]
    pi = [1] * K
    sigma = [np.identity(n_col) for i in range(K)]
    theta = np.array([np.repeat(1., n_row*K)]).reshape((n_row, K))
    
    for i in range(n_iter):

        #E-Step
        for r in range(n_row):
            x = data[r]
            temp_theta = [None] * K
            for k in range(K):
                temp_theta[k] = theta[r, k] * sp.stats.multivariate_normal.pdf(x, mu[k], sigma[k])
            
            theta[r] = temp_theta/ np.sum(temp_theta)[np.newaxis]
        
        #M-Step
        pi = np.mean(theta, axis = 0)
        mu = np.dot(theta.T, data) / np.sum(theta, axis = 0)[:,np.newaxis]
        
        for k in range(K):
            x = data - mu[k]
            theta_diag = np.matrix(np.diag(theta[:, k]))
            temp_sigma = x.T * theta_diag * x
            sigma[k] = temp_sigma / np.sum(theta, axis = 0)[k]
        
        filename = "pi-" + str(i+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(i+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",") 
        
        for j in range(K):
            filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv"
            np.savetxt(filename, sigma[j], delimiter=",")
    

if __name__ == '__main__':

    data = np.genfromtxt("X_train.csv", delimiter = ",")
    K = 5
    n_iter = 10
    
    # Run K-Means
    KMeans(data, K, n_iter)

    # Run EM for GMM
    EMGMM(data, K, n_iter)