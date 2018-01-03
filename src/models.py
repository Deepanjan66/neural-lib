import numpy as np
import random
from error_evals import *


class LayerConnections:
    def __init__(self, dims):
        self.weight_matrix = np.ones((dims['output'],dims['input']))
        self.biases = np.zeros(dims['output'], order="C")

class Neuron:
    def __init__(self, value=0):
        self.value = value
    def update_value(self, value):
        self.value = value
    def get_value(self):
        return self.value



class Layer:
    def __init__(self, num_neurons, activation=None):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.connections = None
        self.activation = activation

    def connect_new_layer(self, layer):
        self.connections = LayerConnections({'input':len(layer.neurons),'output':len(self.neurons)})
    
    def feed_forward(self, inputs):
        
        output_vec = np.matmul(self.connections.weight_matrix, np.transpose(inputs)) + self.connections.biases
        if self.activation:
            output_vec = [self.activation.evaluate(out) for out in output_vec]
        
        # This for loop goes through the list of neurons and
        # updates their values with the values provided 
        # through the new inputs
        for i in range(len(output_vec)):
            self.neurons[i].update_value(output_vec[i])

        return output_vec

    def get_neuron_values(self):
        return [neuron.get_value() for neuron in self.neurons]
    
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
        outputs = np.array(inputs).transpose()
        for layer in self.layers[1:]:
            outputs = layer.feed_forward(outputs)

        print(outputs)

    def correct(self, outputs, target_outputs):
        for in_layer, out_layer in zip(reversed(self.layers[1:]), reversed(self.layers[:-1])):
            out_layer.backprop_error(in_layer.get_neuron_values())


out - target






