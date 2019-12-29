from __future__ import division
import numpy as np
import sys

def PMF(data, user_n, obj_n, n_iter, lmbd, sigma2, d):
    """ Probabilistic Matrix Factorization algorithm
    
    Args:
        data: Numerical non-labelled data matrix (rating matrix)
        user_n: Number of users (observations)
        obj_n: Number of objects (features)
        n_iter: Number of iterations
        lmbd: Lambda L2 regularization parameter
        sigma2: Variance
        d: Number of latent features
    
    Returns: Returns the log join likelihood at each iteration as well as the MAP solution for the matrices U and V
    
    """
    
    length = data.shape[0]
    Nu = user_n
    Nv = obj_n

    #Format input rating matrix
    measured = np.zeros((Nu, Nv))
    ratings = np.zeros((Nu, Nv))

    for k in range(length):
        i = int(data[k, 0]) - 1
        j = int(data[k, 1]) - 1
        measured[i, j] = 1
        ratings[i, j] = data[k, 2]
        
    ##initialize locations and users
    L = np.zeros(n_iter)
    U_matrices = np.zeros((n_iter, Nu, d))
    V_matrices = np.zeros((n_iter, Nv, d))
    
    #initialize V as multivariate gaussian
    mean = np.zeros(d)
    cov = (1/float(lmbd))*np.identity(d)
    V_matrices[0, :, :] = np.random.multivariate_normal(mean, cov, Nv) 

    for k in range(n_iter):
        print('Iteration: ', k+1, ' / ', n_iter)

        ##update user location
        if k == 0:
            l = 0
        else:
            l = k-1

        for i in range(Nu):
            A = lmbd * sigma2 * np.identity(d)
            vec = np.zeros(d)
            for j in range(Nv):
                if measured[i, j] == 1:
                    A += np.outer(V_matrices[l, j, :], V_matrices[l, j, :])
                    vec += ratings[i, j]*V_matrices[l, j, :]
            U_matrices[k, i, :] = np.dot(np.linalg.inv(A), vec)

        ##update object location
        for j in range(Nv):
            A = lmbd * sigma2 * np.identity(d)
            vec = np.zeros(d)
            for i in range(Nu):
                if measured[i, j] == 1:
                    A += np.outer(U_matrices[k, i, :], U_matrices[k, i, :])
                    vec += ratings[i, j]*U_matrices[k, i, :]
            V_matrices[k, j, :] = np.dot(np.linalg.inv(A), vec)

        ##update objective function
        for i in range(Nu):
            for j in range(Nv):
                if measured[i, j] == 1:
                    L[k] -= np.square(ratings[i, j] - np.dot(U_matrices[k, i, :].T, V_matrices[k, j, :]))
        L[k] = (1/(2*sigma2))*L[k]
        L[k] -= (lmbd/float(2))*(np.square(np.linalg.norm(U_matrices[k, :, :])) + np.square(np.linalg.norm(V_matrices[k, :, :])))

        
    return L, U_matrices, V_matrices

if __name__ == "__main__":

    data = np.genfromtxt(sys.argv[1], delimiter = ",")

    lmbd = 2
    sigma2 = 0.1
    d = 5
    dim = (len(np.unique(data[:, 0])), len(np.unique(data[:, 1])))

    # Run PMF function, assuming it returns Loss L, U_matrices and V_matrices
    L, U_matrices, V_matrices = PMF(data, user_n = dim[0], obj_n = dim[1], n_iter = 50, lmbd = lmbd, sigma2 = sigma2, d = d)

    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
