import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        
        self.model = [self.fc1, self.relu, self.fc2]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer in self.model:
            layer.reset_grad()
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        x = X.copy()
        for layer in self.model:
            x = layer.forward(x)
        
        loss, d_out = softmax_with_cross_entropy(x, y)
        
        for layer in reversed(self.model):
            d_out = layer.backward(d_out)
            
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for key in self.params():
            l2_loss, l2_grad = l2_regularization(self.params()[key].value, self.reg)
            loss += l2_loss
            self.params()[key].grad += l2_grad
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = X
        for layer in self.model:
            pred = layer.forward(pred)
        pred = pred.argmax(axis = 1)
        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {'W1': self.fc1.W, 'B1': self.fc1.B, 'W2': self.fc2.W, 'B2': self.fc2.B}

        return result
