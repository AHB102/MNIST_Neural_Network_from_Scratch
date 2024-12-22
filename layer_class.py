import numpy as np
from nn_functions import ReLU_derivative

class Layer:
    def __init__(self, input_size, num_neurons, activation_fxn):
        self.weights = np.random.randn(num_neurons, input_size) * 0.01
        self.bias = np.zeros((num_neurons, 1))  # Initialized to zeros
        self.activation_fxn = activation_fxn

    def forward_propagation(self, inputs):
        self.a_prev = inputs  # Store inputs for backpropagation
        self.z = np.dot(self.weights, inputs) + self.bias  # Linear transformation
        self.a = self.activation_fxn(self.z)  # Apply activation function
        return self.a

    def gradient_calc(self, dl_dy_pred, weights_next, z, a_prev):
        """
    dl_dy_pred: Gradient of the loss with respect to the current layer's output
    weights_next: Weights of the next layer (used for propagating gradients)
    z: Pre-activation outputs of the current layer
    a_prev: Activations from the previous layer (input to the current layer)
     """
    # Compute gradient of loss with respect to the current layer's pre-activation output
        self.dl_dz = dl_dy_pred * ReLU_derivative(z)  # Element-wise multiplication

    # Compute gradients for the weights and biases
        dl_dw = np.dot(self.dl_dz, a_prev.T)  # Gradient w.r.t. weights
        dl_db = np.sum(self.dl_dz, axis=1, keepdims=True)  # Gradient w.r.t. biases

    # Compute the gradient to pass to the previous layer
        dl_da_prev = np.dot(self.weights.T, self.dl_dz)  # Gradient to pass to the previous layer

        return dl_da_prev, dl_dw, dl_db
