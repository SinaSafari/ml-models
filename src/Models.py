import numpy as np


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

    def _predict():
        raise NotImplementedError()

    def _approximation():
        raise NotImplementedError()


class LinearRegression(BaseRegression):

    def _approximation():
        pass

    def _predict():
        pass


class LogisticRegression(BaseRegression):
    pass
