from __future__ import division
import numpy as np
import sys

def paramsMLE(X, y):
    """ Maximum Likelihood parameter estimation function

    Args:
        X: Train data matrix containing explanatory variables
        y: Train data vector containing the response variable 

    Returns: Returns an array containig estimated mu and sigma

    """
    mu = np.empty(shape = (int(max(y) + 1), X.shape[1]))
    S = np.empty(shape = (int(max(y) + 1), X.shape[1]))
    
    for val in range(int(max(y))):
        df_sub = X[y == val, :]
        for i in range(X.shape[1]):
            mu[val, i] = df_sub[:,i].mean()
            S[val, i] = df_sub[:,i].std()
        
    return [mu, S]

def estimatePrior(y):
    """ Class Prior estimation function
    
    Args: 
        y: Train data vector containing the response variable
        
    Returns: Returns estimated class prior distribution
    
    """  
    prior_dist = np.empty(shape = (int(max(y) + 1), 1))
    for val in range(int(max(y))):
        prior_dist[val] = sum(y == val)/len(y)
        
    return prior_dist
    
def simPosterior(X_test, y, prior_dist, mu, S):
    """ Class Posterior simulation function
    
    Args:
        X_test: Test data matrix containing explanatory variables
        y: Train data vector containing the response variable 
        prior_dist: Prior distribution
        mu: Bayes Classifier mean vector
        S: Bayes Classifier covariance matrix
    
    Returns: Returns simulated class posterior distribution
    
    """
    post_dist = np.empty(shape = (y, 1))
    
    for cl in range(y):
        post_cl = 1
        for i in range(x_test.shape[0]):
            rand = np.random.normal(mu[cl, i], S[cl, i], 10000)
            dist = abs(x_test[i] - mu[cl, i])
            lower = rand > mu[cl, i] - dist 
            upper = rand < mu[cl, i] + dist
            post_cl *= (10000 - sum(lower*upper))/10000
        
        if prior_dist[cl] == 0:
            div = 1
        else:
            div = prior_dist[cl]
            
        post_dist[cl] = post_cl/div
        
    return post_dist
    
def pluginClassifier(X_train, y_train, X_test):    
    """ Plug-in Classifier function
  
    Args:
        X_train: Train data matrix containing the explanatory variable 
        y_train: Train data vector containing the response variable 
        X_test: Test data matrix containing explanatory variables
    
    Returns: Returns estimated class posteriors for test data
    
    """
    mu, S = paramsMLE(X = X_train, y = y_train)
    prior_dist = prior(y = y_train)
    
    for i in range(X_test.shape[0]):
        post_dist_i = simPosterior(x_test = X_test[i], y = int(max(y_train) + 1), prior_dist = prior_dist, mu = mu, S = S)
      
        if i == 0:
            post_dist = post_dist_i.T
        else:
            post_dist = np.append(post_dist, post_dist_i.T, axis = 0)

    return post_dist
 
if __name__ == "__main__":

    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")
    
    # Run Plug-in Classifier
    final_outputs = pluginClassifier(X_train, y_train, X_test)

    # Write Plug-in Classifier output to file
    np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file
    