import numpy as np

x = np.array([[8, 0, 1, 1],
              [9, 2, 9, 4],
              [1, 5, 9, 9],
              [9, 9, 4, 7],
              [6, 9, 8, 9]])

# what is a vector y so that y.T * X  is the fourth row of X?
def part_a():
    y = np.array([0, 0, 0, 1, 0])
    return np.matmul(y.T, x)

print(part_a()) # outputs [9 9 4 7]

# given k in {1, 2, 3, 4, 5} construct a vector y st y.T * X is kth row of X
def part_b(k):
    y = np.array([0 if i != k - 1 else 1 for i in range(1, 6)])
    return np.matmul(y.T, x)

print(part_b(1)) # outputs [8 0 1 1]
print(part_b(2)) # outputs [9 2 9 4]
print(part_b(5)) # outputs [6 9 8 9]

# construct vector st y.T * X is a * kth row + b * jth row
def part_c(a, b, j, k):
    if j == k:
        y = np.array([0 if i != j else a + b for i in range(1, 6)])
    else:
        y = np.array([a if i == k else b if i == j else 0 for i in range(1, 6)]) 
    return np.matmul(y.T, x)

print(part_c(2, 3, 1, 2)) # outputs [42 4 21 11]
print(part_c(1, 0, 0, 4)) # outputs [9 9 4 7]
print(part_c(1, 1, 4, 4)) # outputs [18 18  8 14]


# what is a vector w s.t X * w is third column of X?
def part_d():
    w = np.array([0, 0, 1, 0])
    return np.matmul(x, w)

print(part_d()) # outputs [1 9 9 4 8]


# construct vector w s.t X * w is kth column of X
def part_e(k):
    w = np.array([0 if i != k else 1 for i in range(1, 5)])
    return np.matmul(x, w)

print(part_e(1)) # outputs [8 9 1 9 6]
print(part_e(2)) # outputs [0 2 5 9 9]

def part_f(a, b, j, k):
    arr = [0 for i in range(4)]
    arr[k - 1] += a
    arr[j - 1] += b
    w = np.array(arr)
    return np.matmul(x, w)

print(part_f(1, 0, 0, 1)) # outputs [8 9 1 9 6]
print(part_f(2, 3, 4, 4)) # outputs [5 20 45 35 45]

