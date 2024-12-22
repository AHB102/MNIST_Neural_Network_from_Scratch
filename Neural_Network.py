import numpy as np
import pandas as pd
from nn_functions import ReLU, Softmaxfxn, cross_entropy_loss, loss_gradient
from layer_class import Layer
from sklearn.metrics import accuracy_score, classification_report

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def backward_pass(self, expected, actual, learning_rate):
        # Compute the gradient of the loss with respect to the output
        loss_grad = loss_gradient(expected, actual)

        # Iterate through layers in reverse order (backpropagation)
        for layer in reversed(self.layers):
            # Call gradient_calc with the current layer's weights
            loss_grad, dl_dw, dl_db = layer.gradient_calc(loss_grad, None, layer.z, layer.a_prev)

            # Update the weights and biases using gradient descent
            layer.weights -= dl_dw * learning_rate
            layer.bias -= dl_db * learning_rate

# Define network structure
input_layer = Layer(784, 128, ReLU)  # Input layer
hidden_layer_1 = Layer(128, 64, ReLU)  # Hidden layer
output_layer = Layer(64, 10, Softmaxfxn)  # Output layer

nn = NeuralNetwork([input_layer, hidden_layer_1, output_layer])

# Training configuration
epochs = 10
learning_rate = 0.01
batch_size = 32

# Load and preprocess the dataset
df = pd.read_csv("train.csv")

# Split dataset into training and testing subsets
train_df = df.head(700)
test_df = df.iloc[700:1400]

# Extract training data
train_inputs = train_df.iloc[:, 1:].values  # Input data
train_labels = train_df.iloc[:, 0].values  # Output labels

# Extract testing data
test_inputs = test_df.iloc[:, 1:].values  # Input data
test_labels = test_df.iloc[:, 0].values  # Output labels

# Normalize input data
train_inputs = train_inputs / 255.0  # Normalize pixel values to [0, 1]
test_inputs = test_inputs / 255.0  # Normalize pixel values to [0, 1]

# One-hot encode the training labels
num_classes = 10
train_expected_outputs = np.zeros((len(train_labels), num_classes))
for idx, label in enumerate(train_labels):
    train_expected_outputs[idx, label] = 1

# Training loop
for epoch in range(epochs):
    for i in range(0, len(train_inputs), batch_size):
        # Load batch data
        inputs = train_inputs[i:i+batch_size].T
        expected_output = train_expected_outputs[i:i+batch_size].T

        # Forward pass
        predicted_output = nn.forward_pass(inputs)

        # Backward pass
        nn.backward_pass(expected_output, predicted_output, learning_rate)

    # Calculate and print loss after each epoch
    total_loss = cross_entropy_loss(train_expected_outputs.T, nn.forward_pass(train_inputs.T))
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Testing phase
test_predictions = nn.forward_pass(test_inputs.T)  # Forward pass
predicted_labels = np.argmax(test_predictions, axis=0)  # Get the predicted labels

# Evaluate the model
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))
