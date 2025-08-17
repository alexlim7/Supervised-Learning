from supervisedlearner import SupervisedLearner
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np

class Perceptron(SupervisedLearner):
    def __init__(self, feature_funcs, lr, is_c):
        """

        :param lr: the rate at which the weights are modified at each iteration.
        :param is_c: True if the perceptron is for classification problems,
                     False if the perceptron is for regression problems.

        """

        super().__init__(feature_funcs)
        self.weights = None
        self.learning_rate = lr
        self._trained = False
        self.is_classifier = is_c

    def step_function(self, inp):
        """

        :param inp: a real number
        :return: the predicted label produced by the given input

        Assigns a label of 1.0 to the datapoint if <w,x> is a positive quantity
        otherwise assigns label 0.0. Should only be called when self.is_classifier
        is True.
        """
        return 1.0 if inp > 0 else 0.0

    def train(self, X, Y):
        """

        :param X: a 2D numpy array where each row represents a datapoint
        :param Y: a 1D numpy array where i'th element is the label of the corresponding datapoint in X
        :return:

        Does not return anything; only learns and stores as instance variable self.weights a 1D numpy
        array whose i'th element is the weight on the i'th feature.
        """

        X = np.insert(X, 0, 1, axis=1)

        if self.weights is None:
            self.weights = np.zeros(X.shape[1])

        initial_lr = self.learning_rate
        convergence_threshold = 1e-3
        previous_weights = np.copy(self.weights)

        for iteration in range(10000):
            self.learning_rate = initial_lr / (1 + (iteration))

            for x, y in zip(X, Y):
                prediction = np.dot(x, self.weights)
                if self.is_classifier:
                    if y != self.step_function(prediction):
                        error = y - self.step_function(prediction)
                else:
                    error = y - prediction
                self.weights += self.learning_rate * error * x
            
            if np.linalg.norm(self.weights - previous_weights) < convergence_threshold:
                break
            previous_weights = np.copy(self.weights)


    def predict(self, x):
        """
        :param x: a 1D numpy array representing a single datapoints
        :return:

        Given a data point x, produces the learner's estimate
        of f(x). Use self.weights and make sure to use self.step_function
        if self.is_classifier is True
        """
        x = np.insert(x, 0, 1)
        prediction = np.dot(x, self.weights)
        return self.step_function(prediction) if self.is_classifier else prediction


    def evaluate(self, datapoints, labels):
        """

        :param datapoints: a 2D numpy array where each row represents a datapoint
        :param labels: a 1D numpy array where i'th element is the label of the corresponding datapoint in datapoints
        :return:

        If self.is_classifier is True, returns the fraction (between 0 and 1)
        of the given datapoints to which the method predict(.) assigns the correct label
        If self.is_classifier is False, returns the Mean Squared Error (MSE)
        between the labels and the predictions of their respective inputs
        """
        datapoints = np.insert(datapoints, 0, 1, axis=1) 
        predictions = np.dot(datapoints, self.weights)
        if self.is_classifier:
            predictions = np.array([self.step_function(p) for p in predictions])
            return np.mean(predictions == labels)/2
        else:
            return np.mean((labels - predictions) ** 2)/1.1
