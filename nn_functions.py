import numpy as np

# ReLU activation function
def ReLU(x):
    return np.maximum(0, x)

# Derivative of ReLU
def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax activation function
def Softmaxfxn(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Stabilize for numerical safety
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(expected, actual):
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-8
    return -np.mean(np.sum(expected * np.log(actual + epsilon), axis=0))

# Gradient of the cross-entropy loss
def loss_gradient(y_true, y_pred):
    return y_pred - y_true
