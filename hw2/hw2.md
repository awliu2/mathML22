## Math ML Pset 2
### Andi Liu
___
## Question 4
```
import scipy.io as sio 
import numpy as np
```
### Part A
```
# load the training data X and the training labels y
matlab_data_file = sio.loadmat('face_emotion_data.mat')
X = matlab_data_file['X']
y = matlab_data_file['y']

# n = number of data points # p = number of features
n, p = np.shape(X)

# Solve the least-squares solution. w is the list of # weight coefficients
w1 = (np.linalg.inv((X.T).dot(X)).dot(X.T)).dot(y)
# print(w1)
w2 = np.linalg.pinv(X).dot(y)
print(f"weight coefficients:\n{w2}\n")
# w = [[ 0.94366942]
#      [ 0.21373778]
#      [ 0.26641775]
#      [-0.39221373]
#      [-0.00538552]
#      [-0.01764687]
#      [-0.16632809]
#      [-0.0822838 ]
#      [-0.16644364]]
```

### Part B
To classify a new image, we would gather the 9 features from the image,
    compute `(w_0 * f_0) + (w_1 * f_1) + ... + (w_8 * f_8)`
    if this value is > 0, we classify as smiling, and if it is < 0, 
    we classify as not smiling.

### Part C
it appears that feature 1 (with weight 0.943), 
                   feature 4 (with weight -0.39), and 
                   feature 3 (with wiehgt 0.266)
    seem to be relatively important as their weights have the 
    greatest magnitude

### Part D
since it appears that features 1, 3, and 4 have the greatest effect,
   I would use just those three features to attempt to classify a face into
   smiling or not.

### Part E, F

```
def cross_validate(X, y):
    errors = []
    for i in range(0, 128, 16):
        # get subset of values    
        sub_X = [X[j] for j in range(len(X)) if not (i <= j <= i + 15)]
        sub_y = [y[j] for j in range(len(y)) if not (i <= j <= i + 15)]
        
        # get holdout vectors and classifications
        holdout_X=X[i: i + 15]
        holdout_y=y[i: i + 15]
        
        # compute weights
        w_i = np.linalg.pinv(sub_X).dot(sub_y)
        sub_error = 0
        sub_total = 0
        # get error rate of each holdout set
        for j in range(len(holdout_X)):
            sub_total += 1
            prediction = holdout_X[j].dot(w_i)
            # if x * y < 0 then x and y are different signs
            if prediction*holdout_y[j] < 0: 
                sub_error += 1
        errors.append(sub_error / sub_total)

    return np.mean(errors)
```
```
print("part (f):")
# get the error rate using all 9 features
print (f"error value using all 9 features: {cross_validate(X, y)}") # outputs 0.05

# try again with just three features (1, 3, 4 in 1-based indexing)
X_3 = X[:,[0, 2, 3]]
print (f"error value using all 3 features: {cross_validate(X_3, y)}") # outputs 0.0833
```
___
## Question 5
```
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
```