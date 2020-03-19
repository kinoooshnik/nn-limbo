import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # print(self.train_X[0])
        # print(X[0])
        # print(self.train_X[0] - X[0])
        # print(np.abs(self.train_X[0] - X[0]))
        # print(np.sum(np.abs(self.train_X[0] - X[0])))
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                dists[i_test][i_train] = np.sum(np.abs(self.train_X[i_train] - X[i_test]))
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        # print(X[0])
        # print(self.train_X)
        # print(np.sum(np.abs(self.train_X[:] - X[0])))
        # print(np.sum(np.abs(self.train_X[:] - X[0]), axis=1))
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dists[i_test] = np.sum(np.abs(self.train_X[:] - X[i_test]), axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # print(X[:, None])
        # print(self.train_X)
        # print(np.sum(np.abs(X[:, None] - self.train_X), axis=2))
        # TODO: Implement computing all distances with no loops!
        dists = np.sum(np.abs(X[:, None] - self.train_X), axis=2, dtype='float32')
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        # print(self.train_X.shape)
        # print(self.train_y.shape)
        # print(dists.shape)
        # for j in range(dists.shape[1]):
        #     print([dists[0][j]], self.train_y[j])
        # print(dists.shape[1])
        # print(dists.shape[0])
        # print(sorted(dists[0]))
        # print(sorted(dists[0])[0:self.k])
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            arr_with_res = [[dists[i, elem], self.train_y[elem]] for elem in range(dists.shape[1])]
            # print(arr_with_res)
            # print(sorted(arr_with_res)[:self.k])
            # print(sorted(arr_with_res)[:3])
            true_cnt = filter(lambda x: x[1] == True, sorted(arr_with_res)[:self.k])
            false_cnt = filter(lambda x: x[1] == False, sorted(arr_with_res)[:self.k])
            true_len = len(list(true_cnt))
            false_len = len(list(false_cnt))
            # print(true_len > false_len)
            if true_len > false_len:
                pred[i] = True
            else:
                pred[i] = False
            # print(pred)
            # print(sorted(arr_with_res)[:2])
            # print("")
            # print(sorted(arr_with_res)[:2][0][1])
        # print(pred)
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        # print(dists.shape)
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            arr_with_res = [[dists[i, elem], self.train_y[elem]] for elem in range(dists.shape[1])]
            # print(arr_with_res)
            arr_with_res = sorted(arr_with_res)[:self.k]
            # print(arr_with_res)
            num_arr = [arr_with_res[i][1] for i in range(len(arr_with_res))]
            cnt = [[num, num_arr.count(num)] for num in range(10)]
            res = sorted(cnt, key=lambda x: x[1])[-1]
            pred[i] = res[0]
            # print("")
            # print(sorted(arr_with_res)[:3])
            # pass
        return pred
