import numpy as np

class ActivationSigmoid:
    def forward(self, inputs):
        """Activation function"""
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output
    
    def derivative(self, x):
        """Activation function derivative"""
        return  x * (1.0 - x)

class ActivationReLU:
    def forward(self, inputs):
        """Activation function"""
        self.output = np.maximum(0, inputs)
        return self.output
    
    def derivative(self, x):
        """Activation function derivative"""
        x[x<=0]=0
        x[x>0]=1
        return x

class ActivationSoftmax:
    def forward(self, inputs):
        """Activation function"""
        exp_values = np.exp(inputs- np.max(inputs, axis=1, keepdims=True)) 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def calculate_delta_error(self, outputs, class_targets):
        """Used to calculate the delta error for a softmax output layer"""
        num_examples = outputs.shape[0]
        delta_scores = outputs
        #Support both class target shapes (1D and 2D)
        if (len(class_targets.shape) == 1):
            delta_scores[range(num_examples), class_targets] -= 1
        elif (len(class_targets.shape) == 2):
            delta_scores -= class_targets
        else:
            print("unknown class targets shape")
        return delta_scores / num_examples
