import numpy as np
import sys

def Ridge(X, y, lmbd):
    """ Ridge Regression: L2 Regularized Least Squares Function 
    
    Args:
        X: Train data matrix containing explanatory variables
        y: Train data vector containing the response variable 
        lmbd: Lambda regularization parameter. Weight for the penalty term
    
    Returns: Returns the Ridge Regression weights vector

    """
    I = np.identity(X.shape[1])
    wRR = np.around(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lmbd * I), X.T), y), 2)

    return wRR

def ActiveLearning(X0, X, y, lmbd, sigma2):
    """ Active Learning function 
    
    Args:
        X0: Test data matrix containing explanatory variables
        X: Train data matrix containing explanatory variables
        y: Train data vector containing the response variable 
        lmbd: Lambda regularization parameter
        sigma2: Covariance matrix of the predictive distribution
    
    Returns: Returns an array containing the row indexes of the first 10 vectors that would be selected from X0 for active learning
    
    """
    I = np.identity(X.shape[1])
    mu = np.dot(np.dot(np.linalg.inv(lmbd * sigma2 * I + np.dot(X.T, X)), X.T), y)
    S = np.linalg.inv(lmbd * I + np.power(np.sqrt(sigma2), -2) * np.dot(X.T, X))
    w = np.random.multivariate_normal(mu, S)

    mu0 = np.dot(X0, mu)
    S0 = np.diag(sigma2 + np.dot(np.dot(X0, S), X0.T))
    a = [0] * 10

    a2 = S0.argsort()[-10:][::-1]

    for i in range(10):
        ind = np.argmax(S0)
        print(ind + 1)
        x0 = X0[ind,]
        x0 = x0.reshape((x0.shape[0], 1))
        y0 = np.dot(x0.T, w)
        
        X = np.append(X, x0.T, axis = 0)
        y = np.append(y, y0, axis = 0)
        mu = np.dot(np.dot(np.linalg.inv(lmbd * sigma2 * I + np.dot(X.T, X)), X.T), y)
        mu0 = np.dot(X0, mu)
        S = np.linalg.inv(lmbd * I + np.power(np.sqrt(sigma2), -2) * np.dot(X.T, X))
        S0 = np.diag(sigma2 + np.dot(np.dot(X0, S), X0.T))        

        a[i] = (ind + 1)

    return a2

if __name__ == "__main__":

    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

    # Run Ridge Regression
    wRR = Ridge(X_train, y_train, lambda_input)

    # Write Ridge Regression output to file
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") 

    # Run Active Learning
    active = ActiveLearning(X_test, X_train, y_train, lambda_input, sigma2_input)

    # Write Active Learning output to file
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",")
