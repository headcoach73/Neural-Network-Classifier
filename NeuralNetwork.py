import numpy as np
from ActivationFunctions import *
from LossCalculators import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import copy

def load_network(filename, regulization, new_name=""):
    """Returns a network object loaded with settings from a file"""
    if (new_name == ""):
        new_name = filename
    network = NeuralNetwork(filename, [1,1], regulization)
    network.load_network(filename +'.npz')
    return network

class NeuralNetwork:
    def __init__(self, name, network_shape, reg,  loss_calc=None):
        self.name = name

        #Create Layers
        self.layers = []
        self.shape = network_shape
        next_input_size = network_shape[0]
        
        for layer_size in network_shape[1:-1]:
            self.layers.append(LayerDense(next_input_size, layer_size, ActivationReLU()))
            next_input_size = layer_size

        #Create output layer with softmax
        self.layers.append(LayerDense(next_input_size, network_shape[-1], ActivationSoftmax()))

        #Set regulization strength and loss calculator
        self.regulization_strength = reg
        if (loss_calc == None):
            self.loss_calculator = CategoricalCrossEntropy(self.regulization_strength)
        else:
            self.loss_calculator = loss_calc

    def predict(self, input):
        """Prints the networks prediction based on a given input"""
        self.evaluate_network(input)
        output_class = np.argmax(self.network_output)
        confidence = self.network_output[0][output_class]
        print(f'Prediction: {output_class} Confidence: {confidence:.2f}')


    def evaluate_network(self, inputs):
        """Evaluates the network for a given input and returns the output, 
        the output is also stored in self.network_output"""
        self.inputs = inputs
        next_input = inputs
        for layer in self.layers:
            layer.forward(next_input)
            next_input = layer.output
        self.network_output = next_input
        return self.network_output

    def train(self, inputs, class_targets, learning_rate, epoch, batch_size, update_rate =10, show_fig = False, tolerance = 2e-7):
        """Trains the network using the given settings"""
        for epoch in range (epoch):
            #Shuffle training data every epoch
            inputs, class_targets = shuffle(inputs,class_targets)
            total_loss = 0
            total_accuracy = 0
            total_batches = int(inputs.shape[0]/batch_size) + 1
            for n in range(total_batches):
                #Calculate correct slice to create batch from
                lower_bound = (n*batch_size) 
                upper_bound = min((n+1)*batch_size, inputs.shape[0])

                input_batch = inputs[lower_bound:upper_bound]
                target_batch = class_targets[lower_bound:upper_bound]

                self.evaluate_network(input_batch)
                total_loss += self.calculate_loss(target_batch)
                total_accuracy += self.calculate_accuracy(target_batch)
                self.back_propergate(target_batch)
                self.update_weights(learning_rate)
            
            #Print an update on training progress
            if ((epoch+1) % update_rate == 0):
                print(f"Epoch: {epoch+1}")
                print(f"Loss Value: {total_loss/total_batches}")
                print(f"Accuracy: {total_accuracy/total_batches}")
                print("")
                if (show_fig):
                    self.plot_classifier(inputs,class_targets)

            #Stop training if loss below threshold
            if (total_loss/total_batches <= tolerance):
                    return
    
    def set_loss(self, loss_calculator):
        """Change loss calculator object"""
        self.loss_calculator = loss_calculator

    def back_propergate(self, class_targets):
        """Back propergates the delta errors used for updating weights,
        should only be called from train"""
        output_layer = self.layers[-1]
        output_layer.delta_errors = output_layer.activation.calculate_delta_error(output_layer.output, class_targets)
        layer_above = output_layer
        for i in reversed(range(len(self.layers)-1)):
            self.layers[i].back_propergate_error(layer_above)
            layer_above = self.layers[i]

    def update_weights(self, learning_rate):
        """Updates weights and biases of the layers in the network.
        Requires delta errors to already calculated by back_propergate.
        Should only be called from train"""
        inputs = self.inputs
        for layer in self.layers[:-1]:
            layer.update_weights(learning_rate, inputs, self.regulization_strength)
            inputs = layer.output
        self.layers[-1].update_weights(learning_rate, inputs, self.regulization_strength, is_output_layer=True)

    def calculate_loss(self, class_targets):
        """Calculates loss from given class_targets using the current network_output. Should be used after evaluate_network"""
        self.loss = self.loss_calculator.calculate(self.network_output, class_targets, self)
        return self.loss

    def calculate_loss_from_data(self, inputs, class_targets):
        """Calculates loss from given class_targets and inputs"""
        self.evaluate_network(inputs)
        return self.calculate_loss(class_targets)

    def calculate_accuracy(self, class_targets):
        """Calculates accuracy from given class_targets using the current network_output. Should be used after evaluate_network"""
        predictions = np.argmax(self.network_output, axis=1)
        if (len(class_targets.shape) == 2):
            actual = np.argmax(class_targets, axis=1)
        else:
            actual = class_targets
        self.accuracy = np.mean(predictions == actual)
        return self.accuracy

    def save_network(self, save_name =""):
        """Saves the network weights/biases/settings to a npz file. If no save name given the networks name is used as the filename"""
        if (save_name == ""):
            save_name = self.name
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weights)
            biases.append(layer.biases)
        biases.append(np.zeros((2, 1))) #This is needed so np.array doesn't try to make a 3d array
        weights = np.array(weights, dtype=object)
        biases = np.array(biases, dtype=object)
        np.delete(biases, -1)

        np.savez(save_name, Weights=weights, Biases=biases, Shape=self.shape)

    def load_network(self, filename):
        """Loads a npz file containing network weights/biases/settings to update the network settings"""
        LoadOnDemand = np.load(filename, allow_pickle=True)
        network_shape = LoadOnDemand['Shape']
        self.__init__(self.name, network_shape, self.regulization_strength)
        Weights = LoadOnDemand['Weights']
        Biases = LoadOnDemand['Biases']
        for i in range(len(self.layers)):
            self.layers[i].weights = Weights[i]
            self.layers[i].biases = Biases[i]


class LayerDense:
    def __init__(self, n_inputs, n_neurons, activation):
        """Weights is in the form rows=inputs, column=neurons_weights"""
        self.delta_errors = []
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.weights = 0.010 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        
    def forward(self, inputs):
        """Evaluates an input given to the layer including the activation function. Output is in the form rows=outputs, column=neurons"""
        score = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(score)
        return self.output

    def back_propergate_error(self, layer_above):
        """Back propergates error using the given layer above"""
        self.delta_errors = np.dot(layer_above.delta_errors, layer_above.weights.T)
        self.delta_errors *= self.activation.derivative(copy.deepcopy(self.output)) #Deep copy needed so output values are not overridden

    def update_weights(self, learning_rate, inputs, regularization, is_output_layer=False):
        """Updates the layers weights, requires delta_errors to be calculated beforehand"""
        deltaWeights = np.dot(inputs.T, self.delta_errors)
        deltaWeights += regularization*self.weights
        deltaBiases = 0

        #Output layer generally doesn't have biases, this ensures the biases stay at 0 for output layer
        if (not is_output_layer):
            deltaBiases = np.sum(self.delta_errors, axis=0, keepdims=True)

        self.weights += -learning_rate * deltaWeights
        self.biases += -learning_rate * deltaBiases
        