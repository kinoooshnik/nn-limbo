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
        self.fcl1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.act = ReLULayer()
        self.fcl2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        params = self.params()
        for p in params.keys():
            params[p].grad = np.zeros_like(params[p].value)

        out1 = self.act.forward(self.fcl1.forward(X))
        out2 = self.fcl2.forward(out1)

        loss, grad = softmax_with_cross_entropy(out2, y)

        self.fcl1.backward(self.act.backward(self.fcl2.backward(grad)))

        for p in params.keys():
            l2_loss, l2_grad = l2_regularization(params[p].value, self.reg)
            loss += l2_loss
            params[p].grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """
        out2 = self.fcl2.forward(self.act.forward(self.fcl1.forward(X)))
        y_pred = out2.argmax(axis=1)
        return y_pred

    def params(self):
        return {'W1': self.fcl1.W, 'B1': self.fcl1.B, 'W2': self.fcl2.W, 'B2': self.fcl2.B}