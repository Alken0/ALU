# This is a sample Python script.

import numpy as np

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def linear_regression():
    X = np.full((6, 2), [[18, 53.13], [58, 49.03], [23, 17.38], [45, 21], [63, 21.66], [36, 28.50]])
    Y = np.full((6,), [1163.43, 11381.33, 2775, 7222, 14349, 6548])
    inner = X.T @ X
    inversion = np.linalg.inv(inner)
    w = inversion @ X.T @ Y
    return w

def calculate_values(weight, values):
    return weight.T @ values

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    weights = linear_regression()
    test_data = np.full((2,), [18, 53.13])
    test_result = calculate_values(weights, test_data)
    print(test_result)

    data = np.full((2,), [40, 32.5])
    result = calculate_values(weights, data)
    print(result)

