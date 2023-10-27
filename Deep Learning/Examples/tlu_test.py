import numpy as np

epsilon = 0.0000001

def perceptron(x1, x2):
    w1, w2, b = 1.0, 1.0, -1.5
    sum = x1*w1 + x2*w2 + b
    if sum > epsilon:
        return 1
    else:
        return 0

# Numpy 사용 구현

def perceptron_with_numpy(x1, x2):
    X = np.array([x1, x2])
    W = np.array([1.0, 1.0])
    B = -1.5
    sum = np.dot(W, X) + B
    if sum > epsilon:
        return 1
    else:
        return 0