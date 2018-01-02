import numpy as np
import random
from error_evals import *


class LayerConnections:
    def __init__(self, dims):
        self.weight_matrix = np.ones((dims['output'],dims['input']))
        self.biases = np.zeros(dims['output'], order="C")

class Neuron:
    def __init__(self):
        pass



class Layer:
    def __init__(self, num_neurons):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.connections = None

    def connect_new_layer(self, layer):
        self.connections = LayerConnections({'input':len(layer.neurons),'output':len(self.neurons)})
    
    def feed_forward(self, inputs):
        return np.matmul(self.connections.weight_matrix, np.transpose(inputs)) + self.connections.biases
        
    def __repr__(self):
        return "Layer with " + str(len(self.neurons)) + " neurons"


class Graph:
    def __init__(self, num_inputs, error_func=SquaredSum(), learning_rate=0.9):
        self.layers = [Layer(num_inputs)]
        self.num_layers = 1
        self.learning_rate = learning_rate
        self.error_func = error_func

    def add(self, new_layer, weights=None):
        self.layers.append(new_layer)
        self.num_layers += 1
        new_layer.connect_new_layer(self.layers[-2])

    def __repr__(self):
        return "Network with " + str(self.num_layers) + " layers.\n" + \
                "\n".join([str(layer) for layer in self.layers])

    def feed(self, inputs):
        inputs = np.array(inputs).transpose()
        for layer in self.layers[1:]:
            inputs = layer.feed_forward(inputs)

        print(inputs)

