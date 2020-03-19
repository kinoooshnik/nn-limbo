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
        self.relul = ReLULayer()
        self.fully_connect_1 = FullyConnectedLayer(n_input, hidden_layer_size) 
        self.fully_connect_2 = FullyConnectedLayer(hidden_layer_size, n_output)

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
        for i in self.params():
            self.params()[i] = np.zeros_like(self.params()[i].value)
           
        out1 = self.relul.forward(self.fully_connect_1.forward(X))
        out2 = self.fully_connect_2.forward(out1)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss, grad = softmax_with_cross_entropy(out2, y)

        d_out2 = self.fully_connect_2.backward(grad)
        self.fully_connect_1.backward(self.relul.backward(d_out2))

        for param in self.params():
            l2_loss, l2_grad = l2_regularization(self.params()[param].value, self.reg)
            loss += l2_loss
            self.params()[param].grad += l2_grad

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
        pred = np.zeros(X.shape[0], np.int)
        out1 = self.relul.forward(self.fully_connect_1.forward(X))
        out2 = self.fully_connect_2.forward(out1)
        pred = out2.argmax(axis=1)

        return pred

    def params(self):
        result = {'W1': self.fully_connect_1.W, 'B1': self.fully_connect_1.B, 'W2': self.fully_connect_2.W, 'B2': self.fully_connect_2.B}

        # TODO Implement aggregating all of the params

        return result
