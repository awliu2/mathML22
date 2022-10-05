import numpy as np
import pandas as pd



expenses = np.array([[2500, 2000, 2000, 2000],
                [350, 405, 325, 210],
                [200, 250, 400, 450]])

v3 = np.array([1,1,1])
v4 = np.array([1,1,1,1])


# function to get monthly expenses
def get_monthly():
    return np.matmul(expenses.T, v3)

print(get_monthly()) # outputs [3050 2655 2725 2660]


# function to get categorical costs over the 4 months
def get_categories():
    return np.matmul(expenses, v4)

print(get_categories()) # outputs [8500 1290 1300]


# function to get total expenditure
def get_total_spending():
    return np.matmul(np.matmul(expenses, v4).T, v3)

print(get_total_spending()) # outputs 11090
