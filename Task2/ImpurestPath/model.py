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
        self.layers = [
          FullyConnectedLayer(n_input, hidden_layer_size),
          ReLULayer(),
          FullyConnectedLayer(hidden_layer_size, n_output)
        ]
        

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

        # def clear_grad(param):
        #       param.grad = np.zeros_like(param.grad)
        #       print('Clear')
        # map(clear_grad, self.params().items())

        for _, param in self.params().items():
              param.grad = np.zeros_like(param.grad)

        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        for layer in self.layers:
              X = layer.forward(X)
        
        loss, grad = softmax_with_cross_entropy(X, y)
        for i in range(len(self.layers)):
              grad = self.layers[len(self.layers) - 1 - i].backward(grad)
        # After that, implement l2 regularization on all params
        for _, param in self.params().items():
              r_loss, r_grad = l2_regularization(param.value,self.reg)
              loss += r_loss
              param.grad += r_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0])
        for layer in self.layers:
              X = layer.forward(X)
        pred = X.argmax(axis=1)

        return pred

    def params(self):
        result = {
          'W1': self.layers[0].params()['W'],
          'B1': self.layers[0].params()['B'],
          'W2': self.layers[2].params()['W'],
          'B2': self.layers[2].params()['B']
        }

        

        return result
