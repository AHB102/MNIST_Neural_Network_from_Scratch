import numpy as np
import pandas as pd

from nn_functions import Softmaxfxn
from nn_functions import ReLU
from nn_functions import loss_function

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randint(0,1,input_size)
        self.bias = np.random.randint(0,1)

    def forward_propagation(self, input):
        current = 0
        for i in range(len(self.weights)):
            current += np.multiply(self.weights[i], input[i])
        
        current = current + self.bias
        return ReLU(current)


df = pd.read_csv("train.csv")
input = input = df.iloc[0, 2:]
input = np.array(input)

expected_output = np.zeros(10)
expected_output[df.loc[0, df.columns[0]]] = 1

# print(expected_output)

# Hidden layer 1 : 3 Neurons
hidden_layer_1 =[]

# Neuron 1
n11 = Neuron(len(input))
hidden_layer_1.append(n11.forward_propagation(input))
# Neuron 2
n12 = Neuron(len(input))
hidden_layer_1.append(n12.forward_propagation(input))
# Neuron 3
n13 = Neuron(len(input))
hidden_layer_1.append(n13.forward_propagation(input))


# Hidden layer 2 : 3 Neurons
hidden_layer_2 =[]

# Neuron 1
n21 = Neuron(len(hidden_layer_1))
hidden_layer_2.append(n21.forward_propagation(hidden_layer_1))
# Neuron 2
n22 = Neuron(len(hidden_layer_1))
hidden_layer_2.append(n22.forward_propagation(hidden_layer_1))
# Neuron 3
n23 = Neuron(len(hidden_layer_1))
hidden_layer_2.append(n23.forward_propagation(hidden_layer_1))


#Output Layer 
output = []

f0 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f0.forward_propagation(hidden_layer_2)))
f1 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f1.forward_propagation(hidden_layer_2)))
f2 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f2.forward_propagation(hidden_layer_2)))
f3 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f3.forward_propagation(hidden_layer_2)))
f4 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f4.forward_propagation(hidden_layer_2)))
f5 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f5.forward_propagation(hidden_layer_2)))
f6 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f6.forward_propagation(hidden_layer_2)))
f7 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f7.forward_propagation(hidden_layer_2)))
f8 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f8.forward_propagation(hidden_layer_2)))
f9 = Neuron(len(hidden_layer_2))
output.append(Softmaxfxn(f9.forward_propagation(hidden_layer_2)))


print(loss_function(expected_output, output))

