import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


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
    predictions_copy = predictions.copy()
    if len(predictions_copy.shape) == 1:
        predictions_copy -= np.max(predictions_copy)
        return np.exp(predictions_copy) / sum(np.exp(predictions_copy))
    else:
        predictions_copy = np.apply_along_axis(lambda x: x - np.max(x), 1, predictions_copy)
        return np.apply_along_axis(lambda x: np.exp(x) / sum(np.exp(x)), 1, predictions_copy)


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
    if len(probs.shape) == 1:
        return -np.log(probs)[target_index]
    else:
        return np.mean(-np.log([probs[i, x] for i, x in enumerate(target_index.flatten())]))


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO copy from the previous assignment
    sm = softmax(predictions)
    loss = cross_entropy_loss(sm, target_index)
    dprediction = sm

    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        for i, x in enumerate(target_index):
            dprediction[i, x] -= 1
        dprediction /= predictions.shape[0]
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.bin = (X > 0).astype(int)
        return X * self.bin

    def backward(self, d_out):
        # TODO copy from the previous assignment
        return d_out * self.bin

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        self.W.grad += self.X.T.dot(d_out)
        self.B.grad += np.sum(d_out, axis=0)
        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = np.pad(X, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)))

        filter_size, filter_size, input_channels, output_channels = self.W.value.shape
        W = self.W.value.reshape((filter_size * filter_size * input_channels, output_channels))

        out_height = height + 2 * self.padding - filter_size + 1
        out_width = width + 2 * self.padding - filter_size + 1

        result = np.zeros((batch_size, out_height, out_width, output_channels))
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                I = self.X[:, y : y + filter_size, x : x + filter_size, :] \
                    .reshape((batch_size, filter_size*filter_size*input_channels))
                result[:, y, x] = I @ W + self.B.value
        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        filter_size, filter_size, input_channels, output_channels = self.W.value.shape

        result = np.zeros_like(self.X)

        W = self.W.value.reshape((filter_size * filter_size * input_channels, output_channels))

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                result[:, y : y + filter_size, x : x + filter_size, :] += \
                    (d_out[:, y, x, :] @ W.T) \
                    .reshape((batch_size, filter_size, filter_size, input_channels))
                
                I = self.X[:, y : y + filter_size, x : x + filter_size, :] \
                    .reshape((batch_size, filter_size*filter_size*input_channels))
                self.W.grad += (I.T @ d_out[:, y, x, :]).reshape(self.W.value.shape)

                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)

        if self.padding != 0:
            return result[:, self.padding : - self.padding, self.padding : - self.padding, :]
        else:
            return result

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height, out_width = np.ceil([
            (height - self.pool_size) / self.stride,
            (width -  self.pool_size) / self.stride
        ]).astype(int) + 1
        result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                current = X[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :]
                result[:, y, x, :] = np.max(current, axis=(1, 2))
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        _, out_height, out_width, _ = d_out.shape
        batch_size, height, width, channels = self.X.shape
        
        result = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                current_X = self.X[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :]
                mask = (current_X == np.max(current_X, axis=(1,2))[:, np.newaxis, np.newaxis, :])
                result[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :] += grad * mask

        return result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = self.X_shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, height * width * channels))

    def backward(self, d_out):
        batch_size, height, width, channels = self.X_shape
        # TODO: Implement backward pass
        return d_out.reshape((batch_size, height, width, channels))

    def params(self):
        # No params!
        return {}
