import numpy as np
import random
import time

from error_evals import *
import error_handler


class LayerConnections:
    """
    This class creates a matrix which will contain all
    the weights of the connections between layers in the 
    network. This will be used to perform matrix operations
    optimally.

    For example: output layer     0 0 
                 input layer    0  0  0
                 The matrix formed will look like the following
                    0    1    2
                  ===============
                0 |  1 |  1 | 1 |
                  ===============
                1 |  1 |  1 | 1 |
                  ===============
                 
    """
    def __init__(self, dims):
        """
        :param dims: A dictionary that contains the number of 
                     input and output neurons in the layer.
        """
        self.weight_matrix = np.ones((dims['output'],dims['input']))
        self.biases = np.zeros(dims['output'], order="C")

class Neuron:
    """
    This class will represent individual neurons in the network
    """
    def __init__(self, value=0):
        """
        :param value: Value here represents the input this neuron
                      receives during forward pass
        """
        self.value = value
    def update_value(self, value):
        """
        Function for updating the value of the neuron
        :param value: Value here represents the input this neuron
                      receives during forward pass
        """
        self.value = value
    def get_value(self):
        """
        Function for retrieving the value of the neuron
        """
        return self.value

class Layer:
    """
    This class will be used to represent a layer in the neural network.
    """
    def __init__(self, num_neurons, activation=None, learning_rate=None, bias=True):
        """
        :param num_neurons: Number of neurons in the layer
        :param activation: Instance of the activation function that will be used 
                           in this layer
        :param learning_rate: Learning rate to be used for this layer. If not 
                              provided, the default global learning rate
                              from the gtaph will be used
        :param bias: Optional argument for including bias vectors in this layer.
                     Bias vector will have no effect if bias is False
        """
        # creates neuron instances for the layer
        # Using a iD matrix would be better for storing the inputs 
        # but a neuron class makes the code more readable
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.connections = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.bias = bias
        if activation:
            # Vectorizes the derivative function of the activation for optimized 
            # calculations later
            self.activation_der = np.vectorize(self.activation.derivative, otypes=[np.float])

    def connect_new_layer(self, layer):
        """
        This function creates a weight matrix to keep track 
        of the weights of the connections between the layers
        :param layer: Previous layer in the network
        """
        self.connections = LayerConnections({'input':len(layer.neurons),'output':len(self.neurons)})
    
    def feed_forward(self, inputs):
        """
        :oaram inputs: Inputs to be fed into the network
        :return output_vec: The output obtained from this layer
        """
        output_vec = np.matmul(self.connections.weight_matrix, np.transpose(inputs)) + self.connections.biases
        if self.activation:
            output_vec = [self.activation.evaluate(out) for out in output_vec]
        
        self.update_neuron_values(output_vec)

        return output_vec
    
    def update_neuron_values(self, vec):
        """
        :param vec: Vector containing the new inputs of the neurons
        """
        for i in range(len(vec)):
            self.neurons[i].update_value(vec[i])

    def get_neuron_values(self):
        """
        Function for retrieving neuron values
        """
        return np.array([neuron.get_value() for neuron in self.neurons])

    def backprop_error(self, errors, input_array):
        """
        Method for backpropagating error through the network
        and updating the weights.

        This method has been optimized to only used matrix operations
        with numpy to reduce the time required for training with backprop

        :param errors: A 1D matrix containg the errors from the previous layer
        :param input_array: An array containing the inputs received by the layer
                            during the forward pass. This will be required for
                            the derivatives
        """
        # Get the weights associated with the connections between this
        # layer and the layer underneath it
        weight_matrix = self.connections.weight_matrix
        bias_vector = self.connections.biases
        # The goal here is to use matrix operations to perform the whole backprop
        # process.
        # This process will be outlined in the blog associated with this repo
        # but a quick description is given below
        # The formula for updating the weights is: w = w - (eta)*(dE/dw)
        # where (eta) is the learning rate, w is the weight to be updated
        # E is the error obtained from the error function
        # The `errors` list contains the errors from the previos layer
        #
        # First the input list is repeated to match the size of the weight matrix
        updated_errors = np.repeat([input_array], weight_matrix.shape[0], axis=0)
        # The error from each output neuron is the multiplied to each row of the 
        # inputs
        updated_errors = updated_errors * (self.learning_rate * errors[:, np.newaxis])
        # If an activation function is provided, that function is mapped to
        # all the inputs
        if self.activation:
            updated_errors = self.activation_der(updated_errors)
        
        # Subtract the error from the current weights
        weight_matrix -= updated_errors
        # The error update for the bias vector will be the direct error signals 
        # coming from the neurons in the previous layer
        if self.bias:
            # only update the bias if the user wants it. Otherwise, bias 
            # will remain 0 and will have no influence on the network
            bias_vector -= errors
        # Add the error associated with the connections of each neuron
        # and that will be passed on to the next layer
        # This means that the same derivatives or errors aren't calculated twice
        summed_updated_errors = np.sum(updated_errors, axis=0)
        return summed_updated_errors

    def __repr__(self):
        return "Layer with " + str(len(self.neurons)) + " neurons"


class Graph:
    def __init__(self, num_inputs, error_func=SquaredSum(), global_learning_rate=0.9):
        """
        :param num_inputs: Number of input neurons
        :param error_func: provided error function. 
                           Otherwise, defaults to Squared sum
        :param global_learning_rate: Learning rate to be used for layers
                                     that do not have a provided learning rate
        """
        # The input layer is created when the graph is created
        self.layers = [Layer(num_inputs)]
        self.num_layers = 1
        self.global_learning_rate = global_learning_rate
        self.error_func = error_func

    def add(self, new_layer, weights=None):
        """
        This function can be used to add a new layer to the
        network
        :param new_layer: The new layer to be added to the network
        :param weights: Custom weights if required (Not implemented yet)
        """
        self.layers.append(new_layer)
        self.num_layers += 1
        new_layer.connect_new_layer(self.layers[-2])
        if not new_layer.learning_rate:
            new_layer.learning_rate = self.global_learning_rate

    def __repr__(self):
        return "Network with " + str(self.num_layers) + " layers.\n" + \
                "\n".join([str(layer) for layer in self.layers])
    
    def predict(self, inputs):
        """
        This method provides the network's prediction for the
        provided outputs
        :param inputs: List of lists containing all the supplied inputs
        
        :return all_outputs: A list of all outputs produced by the 
                             network
        """
        # check if inputs are lists of lists
        error_handler.check_list_of_lists(inputs)
        # If multiple training data supplied, go through all of them 
        # and train the network
        
        # List for storing all the outputs
        all_outputs = []
        
        for input_data in inputs:
            outputs = np.array(input_data).transpose()
            # The provided inputs are provided to the input layer
            self.layers[0].update_neuron_values(outputs)
            # The output from one layer is passed onto the
            # next layer in the network to complete one complete pass
            for layer in self.layers[1:]:
                outputs = layer.feed_forward(outputs)
            # append outputs to the final list
            all_outputs.append(outputs)

        return all_outputs
        
    def feed(self, inputs, target_outputs, flag=None):
        """
        This function is used to complete one forward pass
        and one backward pass in the training process
        :param inputs: List of inputs to the network
        :param target_output: list of expected outputs
        """
        # check if inputs are lists of lists
        error_handler.check_list_of_lists(inputs, target_outputs)
        # If multiple training data supplied, go through all of them 
        # and train the network
        for input_data, target in zip(inputs, target_outputs):
            outputs = np.array(input_data).transpose()
            # The provided inputs are provided to the input layer
            self.layers[0].update_neuron_values(outputs)
            # The output from one layer is passed onto the
            # next layer in the network to complete one complete pass
            for layer in self.layers[1:]:
                outputs = layer.feed_forward(outputs)
            
            target = np.array(target)
            # Backpropagate the error
            self.correct(outputs, target)

    def correct(self, outputs, target_outputs):
        """
        This function will be used to correct the weights of the network
        :param outputs: Predicted outputs
        :param target_outputs: True outputs
        """
        # Get error from error function
        errors = self.error_func.derivative(outputs, target_outputs)
        for in_layer, out_layer in zip(reversed(self.layers[:-1]), reversed(self.layers[1:])):
            # Propagate the error through the network. The new updated errors from
            # the current layer is returned to be used in the next layer. This
            # ensures that the same derivatives aren't calculated twice.
            errors = out_layer.backprop_error(errors, in_layer.get_neuron_values())


