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
    # DONE: implement softmax
    # Your final implementation shouldn't have any loops

    
    probs = predictions.copy()
    if len(predictions.shape) == 1:
        probs -= np.max(predictions)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    else:
        probs -= np.max(predictions, axis = 1).reshape(-1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs), axis = 1).reshape(-1, 1)
        
        
    return probs

    


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
    # DONE: implement cross-entropy
    # Your final implementation shouldn't have any loops
    if len(probs.shape) == 1:    
        loss = -np.log(probs[target_index])
    else:
        batch_indexes = np.arange(probs.shape[0])
        loss = np.mean(-np.log(probs[batch_indexes,target_index]))

    return loss

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
    # DONE: Copy from previous assignment
    
    loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W


    return loss, grad


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
    # DONE: copy from the previous assignment
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    dprediction = probs.copy()
    
    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        batch_indexes = np.arange(probs.shape[0])
        dprediction[batch_indexes, target_index] -= 1  
        dprediction /= dprediction.shape[0]
    
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
        # DONE: copy from the previous assignment
        out = np.zeros(X.shape)
        out = np.maximum(0, X)
        self.X = X
        return out

    def backward(self, d_out):
        # DONE: copy from the previous assignment

        d_result = d_out * (self.X >= 0)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # DONE: copy from the previous assignment
        
        out = np.dot(X, self.W.value) + self.B.value       
        self.X = X 
        return out

    def backward(self, d_out):
        # DONE: copy from the previous assignment
        
        self.W.grad = np.dot(self.X.T, d_out)    
        self.B.grad = np.sum(d_out, axis=0)[:, np.newaxis].T
        d_input = np.dot(d_out, self.W.value.T)
   
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
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

        out_height = 0
        out_width = 0
        
        # DONE: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
        
        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        #получаем X с padding
        X_padded[:, self.padding:self.padding + height, self.padding:self.padding+width, :] = X

        #сохряняем значения
        self.X_cache = (X, X_padded)
        
        X_padded = X_padded[:, :, :, :, np.newaxis]
        W = self.W.value[np.newaxis, :, :, :, :]

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        
        for y in range(out_height):
            for x in range(out_width):
                # DONE: Implement forward pass for specific location
                
                #выделяем текущий срез X, к которому применяем фильтр
                curr_X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, :]
                #применение фильтра на срез X
                out[:, y, x, :] = np.sum(curr_X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value
                
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        #достаём сохраненные ранее значения
        X, X_padded = self.X_cache
        
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        
        X_grad = np.zeros_like(X_padded)
        
        for y in range(out_height):
            for x in range(out_width):
                # DONE: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                
                #выделяем текущий срез X
                curr_X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * curr_X_slice, axis=0)

                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)
            
            
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]


    def params(self):
        return { 'W': self.W, 'B': self.B }


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
        # DONE: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        #копируем переданные значения X
        self.X = X.copy()
        
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                #выделяем текущий слайс по X
                curr_X_slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                #выбираем и оставляем максимум в выделенной области
                out[:, y, x, :] = np.amax(curr_X_slice, axis=(1, 2))

        return out
        

    def backward(self, d_out):
        # DONE: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                #выделяем текущий слайс по X
                curr_X_slice = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                #получаем значения градиентов
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                #определяем маску
                mask = (curr_X_slice == np.amax(curr_X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                #накладываем маску на значения градиентов
                out[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad * mask

        return out
        
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # DONE: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # DONE: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
