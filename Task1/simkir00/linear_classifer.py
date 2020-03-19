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
    if (predictions.ndim == 1):
        ax = 0
    else:
        ax = 1
        
    probs = np.exp(predictions - np.max(predictions, axis = ax).reshape(-1,1))
    probs = probs / np.sum(probs, axis = ax).reshape(-1, 1)
    
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
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    loss = 0
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        
        # Использовал цикл, потому что без него пришёл в тупик (5 часов поиска вариантов не превели к желаемому результату)
        # Код без циклов должен иметь примерно следующий вид:
        # loss = np.mean(np.sum(-np.log(probs[np.arange(probs.shape[0]), target_index])))
        
        for i in range(probs.shape[0]):
            loss -= np.log(probs[i, target_index[i]])
        loss /= probs.shape[0]
        
    return loss


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    if (probs.ndim == 1):
        probs[target_index] -= 1
    else:
        
        # Также использовал цикл (по причинам описанным выше)
        # Примерный вид кода без цикла:
        # probs[np.arange(probs.shape[0]), target_index] -= 1
        
        for i in range(probs.shape[0]):
            probs[i, target_index[i]] -= 1
        probs /= probs.shape[0]
    dprediction = probs
    
    return loss, dprediction


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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, grad = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, grad)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self, W = None):
        self.W = W
    
    # Функция copy(self) — копирование параметров (весов)
    def copy(self):
        return LinearSoftmaxClassifier(self.W.copy())
        
    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            for batch in batches_indices:
                loss, dW = linear_softmax(X[batch], self.W, y[batch])
                
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
                reg_loss, reg_dW = l2_regularization(self.W, reg)
                self.W -= learning_rate * (dW + reg_dW)
            
            # Смотрим loss по окончании эпохи (но можем смотреть и каждый batch)
            loss_history.append(loss + reg_loss)
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis = 1)
        return y_pred



                
                                                          

            

                
