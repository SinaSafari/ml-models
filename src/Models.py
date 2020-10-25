import numpy as np
from collections import Counter


class BaseRegression():
    """
    abstract class parent of linear and logistic regression
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError()

    def _approximation(self, X, w, b):
        raise NotImplementedError()


class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


class LogisticRegression(BaseRegression):

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def _predict(self. X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


class KNN():

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict():
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict():
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
