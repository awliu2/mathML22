from re import L
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#collaboarated with eric liu


# n = number of points
# z = points where polynomial is evaluated
# p = array to store the values of the interpolated polynomials
n = 100
z = np.linspace(-1, 1, n)
p = []


d = 3 # degree
w = np.random.rand(d) 

X = np.zeros((n,d))



# generate X-matrix
for i in range(n):
    for j in range(3):
        X[i][j] = z[i] ** j

# evaluate polynomial at all points z, and store the result in p # do NOT use a loop for this
p = np.matmul(X, w)

# plot the datapoints and the best-fit polynomials
plt.plot(z, p, linewidth=2)
plt.xlabel('z')
plt.ylabel('y')
plt.title('polynomial_with_coefficients_w_=_%s' %w )
plt.show()