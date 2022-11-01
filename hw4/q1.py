import numpy as np

# function to normalize a vector v
def normal(v):  
    norm = np.linalg.norm(v)
    # if zero vector, do nothing
    if norm < 1e-8: return v
    return v / norm


def gram_Schmidt(X):
    # drop zero columns
    zeros = np.where(~X.any(axis=0))[0]
    for col in zeros:
        X = np.delete(X, col, 1)

    # check that n, p >= 0
    n = len(X)
    assert n > 0
    p = len(X[0])
    assert p > 0
    
    # generate U matrix as transpose as row ops are easier in python
    u_T = np.zeros((p, n))
    # set u_1 (normlaize X_1)
    u_T[0] = normal(X[:,0])

    # x_j' = x_j - sum(U_iX_j)U_i
    for j in range(1, p):
        x_j = X[:,j]
        sum = np.zeros((n, 1))

        # compute sum interatively
        for i in range(j):
            sum = np.add(sum, ((u_T[i] @ (x_j)) * (u_T[i].T)))
        
        # compute x_j' and u_j
        x_j_prime = np.subtract(x_j, sum)
        u_j = normal(x_j_prime)
        
        # set jth row of u_T
        u_T[j,:] = u_j[0]

    # return transpose of U     
    return u_T.T

X = np.array([[0, 2, 3], [3, 12, 1], [4, 1, 3]])
X_2 = np.array([[1,0], [0,3], [0,4]])

print(gram_Schmidt(X)) 
# outputs [[1.         0.        ]
#          [0.         0.34641016]
#          [0.         0.46188022]]

print(gram_Schmidt(X_2))
# outputs [[1.         0.        ]
#          [0.         0.34641016]
#          [0.         0.46188022]]

