import abc
import numpy as np


class Ball:
    """
    Ball(center, radius, label)

    This class is aimed to model a ball of the GEM-balls classifier.

    Parameters
    ----------
    center : array-like
        The center of the sphere.

    radius : float
        The radius of the sphere.

    label : int
        The associated class for this sphere.

    Attributes
    ----------
    center : array-like
        The center of the sphere.

    radius : float
        The radius of the sphere.

    label : int
        The associated class for this sphere.

    """

    def __init__(self, center, radius, label):
        self.center = center
        self.radius = radius
        self.label = label

    def __contains__(self, point):
        return np.linalg.norm(self.center - point) < self.radius


class GEMBallsClassifier:
    """
    GEMBallsClassifier(c0=1, c1=0, algorithm='kd_tree')

    This class implement the GEM-balls classifier.

    For further details about this classifier: http://www.algocare.it/L-CSL2018GEM.pdf

    Parameters
    ----------
    c0 : int, default=1
        During the construction of the model, the radius of spheres computed for points belonging to class 1 is the
        distance between the center and c0-nearest neighbor point belonging to class 0.

    c1 : int, default=1
        During the construction of the model, the radius of spheres computed for points belonging to class 0 is the
        distance between the center and c1-nearest neighbor point belonging to class 1.

    algorithm : {'kd_tree', 'brute'}, default='kd_tree'
        The name of the algorithm to compute the classifier.

    Attributes
    ----------
    c : list of length 2
        The list [c0, c1] with the given c0 and c1.

    k : list of length 2
        The list [k0, k1] with k0 and k1 respectively the support lengths for the classes 0 and 1.
        For the definition of support refer to the paper.

    n : list of length 2
        The list [n0, n1] with n0 and n1 respectively the number of points in the training set belonging to classes 0
        and 1.

    classifier : list of Ball
        List of Ball objects forming the classifier.
        
    """

    algorithm_name = None

    def __new__(cls, algorithm='kd_tree', **kwargs):
        if cls != GEMBallsClassifier and issubclass(cls, GEMBallsClassifier):
            return super(GEMBallsClassifier, cls).__new__(cls)

        implementations = cls.get_algorithms()
        if algorithm not in implementations:
            raise ValueError("Unknown algorithm '%s'" % algorithm)

        subclass = implementations[algorithm]
        instance = super(GEMBallsClassifier, subclass).__new__(subclass)
        return instance

    @classmethod
    def get_algorithms(cls):
        return {subclass.algorithm_name: subclass for subclass in cls.__subclasses__()}

    def __init__(self, c0=1, c1=1, **kwargs):
        self.classifier = []
        self.c = [c0, c1]
        self.k = [0, 0]
        self.n = [0, 0]

    @abc.abstractmethod
    def fit(self, x, y, first=0):
        """
        fit(self, x, y, first=0)

        This method compute the model for the given data using the specified algorithm in the constructor.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            The data of the training set.

        y : array-like of shape (n_queries,)
            The classes associated to x.

        first : int, default=0
            The index of the starting point for the construction of the model.

        """

        raise NotImplementedError("There is no implementation of fit")

    def predict_proba(self, x):
        """
        predict_proba(self, x)

        Estimate the probabilities of belonging to the classes for each target in x.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Target samples.

        Returns
        -------
        y : array of shape (n_queries, 2)
            Estimated probabilities of belonging to class 0 and class 1 respectively for each given target sample.

        Notes
        -----
        This implementation of GEM-balls doesn't compute probabilities.
        This method assign probability of 1 to the predicted class and 0 the the other.

        Examples
        --------
        >>> from gemballs import GEMBallsClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import make_circles
        >>> features, labels = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)
        >>> X_train, X_test, y_train, y_test = \
                 train_test_split(features, labels, test_size=.3, random_state=7)
        >>> gemballs = GEMBallsClassifier()
        >>> gemballs.fit(X_train, y_train)
        >>> estimated_probabilities = gemballs.predict_proba(X_test)
        >>> estimated_probabilities.shape
        (60, 2)

        """
        y = self.predict(x)
        y = y.reshape(-1, 1)
        return np.append(1 - y, y, axis=1)

    def predict(self, x):
        """
        predict(self, x)

        Predict the class label for each target in x.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Target samples.

        Returns
        -------
        y : array of shape (n_queries,)
            Predicted class labels for each given target sample.

        Examples
        --------
        >>> from gemballs import GEMBallsClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import make_circles
        >>> features, labels = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)
        >>> X_train, X_test, y_train, y_test = \
                 train_test_split(features, labels, test_size=.3, random_state=7)
        >>> gemballs = GEMBallsClassifier()
        >>> gemballs.fit(X_train, y_train)
        >>> predicted_classes = gemballs.predict(X_test)

        """

        if not self.classifier:
            raise Exception("You have to train the model before use it")
        if None in x:
            raise ValueError("Dataset contains None value")

        y = np.empty(len(x), dtype=np.int)
        for index, point in enumerate(x):
            for ball in self.classifier:
                if point in ball:
                    y[index] = ball.label
                    break

        return y

    def score(self, x, y):
        """
        score(self, x, y)

        Return the accuracy score of the given test data and labels.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Test samples.

        y : array-like of shape (n_queries,)
            True class of x.

        Returns
        -------
        score : float
            The accuracy of the predicted labels of x with respect to y.

        Examples
        --------
        >>> from gemballs import GEMBallsClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import make_circles
        >>> features, labels = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)
        >>> X_train, X_test, y_train, y_test = \
                 train_test_split(features, labels, test_size=.3, random_state=7)
        >>> gemballs = GEMBallsClassifier()
        >>> gemballs.fit(X_train, y_train)
        >>> accuracy = gemballs.score(X_test, y_test)
        >>> accuracy
        0.8

        """

        if len(x) != len(y):
            raise ValueError("Dataset and labels must have the same length")

        y_predicted = self.predict(x)
        score = np.count_nonzero(y_predicted == np.array(y)) / len(y)
        return score
