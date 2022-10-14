import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# load x and y vectors
d = sio.loadmat('polydata.mat') 
z = d['x']
y = d['y']
# n = number of data points
# N = number of points to use for interpolation
# z = points where interpolant is evaluated
# p = array to store the values of the interpolated polynomials n = z.size
n = z.size
N = 100
z_test = np.linspace(np.min(z), np.max(z), N)

p = np.zeros((3, N))

for d in [1, 2, 3]:
    # generate X-matrix for this choice of d
    X_d = np.zeros((n, (d + 1)))
    for i in range(n):
        for j in range(d + 1):
            X_d[i][j] = z[i] ** j

    # solve least-squares problem. w is the list
    # of polynomial coefficients
    w = np.linalg.pinv(X_d).dot(y)

    # evaluate best -fit polynomial at all points z_test ,
    # and store the result in p
    # p[ = [np.polyval(w, i) for i in z_test]
    for i, val in enumerate(z_test):
        p[d - 1][i] = np.polyval(list(reversed([v[0] for v in w])), val)
    # (optional): this can be done in one line
    # with the polyval command!


# plot the datapoints and the best-fit polynomials
plt.plot(z, y, '.', z_test, p[0, :], z_test, p[1, :], z_test, p[2, :], linewidth=2)
plt.legend(['data', 'd=1', 'd=2', 'd=3'], loc='upper left')
plt.title('best fit polynomials of degree 1, 2, 3')
plt.xlabel('z')
plt.ylabel('y')
plt.show()
 