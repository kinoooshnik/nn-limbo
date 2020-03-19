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
        self.l1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.l2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for k in self.params().keys():
            self.params()[k].grad = np.zeros_like(self.params()[k].grad)

        out = self.l2.forward(self.relu.forward(self.l1.forward(X)))
        loss, grad = softmax_with_cross_entropy(out, y)
        self.l1.backward(self.relu.backward(self.l2.backward(grad)))

        for k in self.params().keys():
            l2_loss, l2_grad = l2_regularization(self.params()[k].value, self.reg)
            loss += l2_loss
            self.params()[k].grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        return self.l2.forward(self.relu.forward(self.l1.forward(X))).argmax(axis=1)

    def params(self):
        return {"W1": self.l1.W,
                "B1": self.l1.B,
                "W2": self.l2.W,
                "B2": self.l2.B}
