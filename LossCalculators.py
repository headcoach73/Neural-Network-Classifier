import numpy as np

class Loss:
    def calculate(self, output, y, network):
        """Calculates the loss of a given output and class targets."""
        sample_losses = self.forward(output, y, network)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossEntropy(Loss):
    def __init__(self,regulization):
        super().__init__()
        self.regulization_strength = regulization

    def forward(self, y_pred, y_true, network):
        """Calculates catergorical cross entropy loss"""
        num_examples = y_pred.shape[0]
        #Clip values away from 0 and 1 because Log(0) is undefined and Log(1) is 0 which doesnt help
        clipped_pred = np.clip(y_pred, 1e-7, 1- 1e-7)

        #Calculates confidences differently based on y_true shape eg [0,2,1] can be given instead of
        #[[1,0,0],
        # [0,0,1],
        # [0,1,0]]
        if len(y_true.shape) == 1:
            correct_confidences = clipped_pred[range(num_examples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(clipped_pred*y_true, axis=1)
        
        correct_confidences_log = -np.log(correct_confidences)
        data_loss = np.sum(correct_confidences_log)/num_examples
        #Calculate regulization 
        reg_loss = 0
        for layer in network.layers:
            reg_loss += 0.5*self.regulization_strength*np.sum(layer.weights*layer.weights)
        
        loss = data_loss + reg_loss
        return loss