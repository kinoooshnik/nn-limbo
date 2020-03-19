import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    # print(predictions.ndim)
    dim = predictions.ndim - 1

    # print(dim)
    max_val = np.max(predictions, axis=dim)

    # print(m.reshape(1, 1))
    # print(m.reshape(-1, 1))
    # print(m.reshape(1, -1))

    if predictions.ndim == 2:
        max_val = max_val.reshape(-1, 1)

    probs = np.exp(predictions - max_val)

    s = np.sum(probs, axis=dim)

    if predictions.ndim == 2:
        s = s.reshape(-1, 1)

    probs = np.divide(probs, s)

    return probs
    # raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    # print([(i, target_index[i]) for i in range(target_index.size)])
    # print(probs)
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        target_index = (np.arange(batch_size), target_index.reshape(batch_size))
        loss = np.mean(-np.log(probs[target_index]))

    # print([[i, probs[i, target_index[i]][0]] for i in range(target_index.size)])
    # print()
    # print(target_index)
    # loss = -np.sum(probs * np.log(probs[target_index]))

    return loss

    # raise Exception("Not implemented!")

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    # raise Exception("Not implemented!")

    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    # raise Exception("Not implemented!")

    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()

    if predictions.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        ti = (np.arange(batch_size), target_index.reshape(batch_size))
        dprediction[ti] -= 1
        dprediction /= batch_size

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        # raise Exception("Not implemented!")
        self.X = X
        return np.maximum(0, X)


    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        # print(d_out * (self.X > 0))
        d_result = d_out * (self.X > 0)

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.B.grad += np.sum(d_out, axis=0).reshape(1, -1)
        self.W.grad += self.X.T.dot(d_out)

        # raise Exception("Not implemented!")

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
