from supervisedlearner import SupervisedLearner
import numpy as np

class KNNClassifier(SupervisedLearner):
    def __init__(self, feature_funcs, k):
        super(KNNClassifier, self).__init__(feature_funcs)
        self.k = k
        self.anchor_points = None
        self.anchor_labels = None

    def train(self, anchor_points, anchor_labels):
        """
        :param anchor_points: a 2D numpy array, in which each row is
						      a datapoint, without its label, to be used
						      for one of the anchor points

		:param anchor_labels: a list in which the i'th element is the correct label
		                      of the i'th datapoint in anchor_points

		Does not return anything; simply stores anchor_labels and the
		_features_ of anchor_points.
		"""
        self.anchor_points = anchor_points
        self.anchor_labels = anchor_labels

    def predict(self, x):
        """
        Given a single data point, x, represented as a 1D numpy array,
		predicts the class of x by taking a plurality vote among its k
		nearest neighbors in feature space. Resolves ties arbitrarily.

		The K nearest neighbors are determined based on Euclidean distance
		in _feature_ space (so be sure to compute the features of x).

		Returns the label of the class to which x is predicted to belong.
		"""
        # A list containing the Euclidean distance of x from another point y,
        # each element of which is in the form (distance, y index)
        # Get the k closest points to x and their labels
        # Note: max(set(x), key=x.count) returns the mode of a list x.
        distances = [np.sqrt(np.sum((x - anchor_point) ** 2)) for anchor_point in self.anchor_points]
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.anchor_labels[i] for i in nearest_indices]
        return max(set(nearest_labels), key=nearest_labels.count)
        

    def evaluate(self, datapoints, labels):
        """
        :param datapoints: a 2D numpy array, in which each row is a datapoint.
		:param labels: a 1D numpy array, in which the i'th element is the
		               correct label of the i'th datapoint.

		Returns the fraction (between 0 and 1) of the given datapoints to which
		predict(.) assigns the correct label
		"""
        # Count the number of correct predictions and find the model accuracy
        correct_predictions = 0
        for i, point in enumerate(datapoints):
            predicted_label = self.predict(point)
            if predicted_label == labels[i]:
                correct_predictions += 1
        return correct_predictions / len(datapoints)
