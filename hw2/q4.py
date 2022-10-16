import scipy.io as sio 
import numpy as np
##### Part a #####
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

"""
b) To classify a new image, we would gather the 9 features from the image,
    compute (w_0 * f_0) + (w_1 * f_1) + ... + (w_8 * f_8)
    if this value is > 0, we classify as smiling, and if it is < 0, 
    we classify as not smiling.

c) it appears that feature 1 (with weight 0.943), 
                   feature 4 (with weight -0.39), and 
                   feature 3 (with wiehgt 0.266)
    seem to be relatively important as their weights have the 
    greatest magnitude

d) since it appears that features 1, 3, and 4 have the greatest effect,
   I would use just those three features to attempt to classify a face into
   smiling or not
"""

# code for part (e) and (f)
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

print("part (f):")
# get the error rate using all 9 features
print (f"error value using all 9 features: {cross_validate(X, y)}") # outputs 0.05

# try again with just three features (1, 3, 4 in 1-based indexing)
X_3 = X[:,[0, 2, 3]]
print (f"error value using all 3 features: {cross_validate(X_3, y)}") # outputs 0.0833



