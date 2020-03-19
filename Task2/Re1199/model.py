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
        self.ReLU1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)


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

        params = self.params()
        for k in params.keys():
            params[k].grad = np.zeros_like(params[k].value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        out = self.fc2.forward(self.ReLU1.forward(self.fc1.forward(X)))

        loss, grad = softmax_with_cross_entropy(out, y)

        self.fc1.backward(self.ReLU1.backward(self.fc2.backward(grad)))

        for k in params.keys():
            l2_loss, l2_grad = l2_regularization(params[k].value, self.reg)
            loss += l2_loss
            params[k].grad += l2_grad

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

        out = self.fc2.forward(self.ReLU1.forward(self.fc1.forward(X)))
        pred = out.argmax(axis=1)

        return pred


    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result['W1'] = self.fc1.W
        result['B1'] = self.fc1.B

        result['W2'] = self.fc2.W
        result['B2'] = self.fc2.B

        return result
