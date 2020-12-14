import numpy as np
from multiprocessing import Pool
from .gemballsclassifier import GEMBallsClassifier


class GEMBallsPollingClassifier:
    """
    GEMBallsPollingClassifier(c0=1, c1=1, max_classifiers=0, algorithm='kd_tree')

    This class implement the polling version of GEM-balls classifier.

    Parameters
    ----------
    c0 : int, default=1
        During the construction of a model, the radius of spheres computed for points belonging to class 1 is the
        distance between the center and c0-nearest neighbor point belonging to class 0.

    c1 : int, default=1
        During the construction of a model, the radius of spheres computed for points belonging to class 0 is the
        distance between the center and c1-nearest neighbor point belonging to class 1.

    max_classifiers : int, default=inf
        Upper bound of the number of classifiers to train.

    algorithm : {'kd_tree', 'brute'}, default='kd_tree'
        The name of the algorithm to compute classifiers.

    Attributes
    ----------
    c : list of length 2
        The list [c0, c1] with the given c0 and c1.

    classifiers : list of GEMBallsClassifier
        List of the trained classifiers.

    max_classifiers : int
        The given parameter max_classifiers.

    algorithm : str
        The given name of the algorithm to use to compute classifiers.

    """

    def __init__(self, c0=1, c1=1, max_classifiers=float('inf'), algorithm='kd_tree'):
        self.classifiers = []
        self.c = [c0, c1]
        self.max_classifiers = max_classifiers
        self.algorithm = algorithm

    def fit(self, x, y, num_of_threads=1):
        """
        fit(self, x, y, num_of_threads=1)

        This method trains many models, at most as many specified in the constructor parameter max_classifiers.
        The models are trained using the algorithm specified in the constructor parameter.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            The data of the training set.

        y : array-like of shape (n_queries,)
            The classes associated to x.

        num_of_threads : int, default=1
            Number of thread to use.

        """

        y = np.array(y)
        x = np.array(x)

        max_classifiers = min(len(x), self.max_classifiers)

        pool = Pool(num_of_threads)
        self.classifiers = pool.starmap(self._fit_single,
                                        [(x, y, i) for i in range(0, max_classifiers)])  # multithreading fit

    def _fit_single(self, x, y, start_index):
        gemballs = GEMBallsClassifier(c0=self.c[0], c1=self.c[1], algorithm=self.algorithm)
        gemballs.fit(x, y, start_index)
        return gemballs

    def predict_proba(self, x, f=lambda k_tot, dim_plus_one: dim_plus_one - k_tot, perc=1, num_of_threads=1):
        """
        predict_proba(self, x, f=lambda k_tot, dim_plus_one: dim_plus_one - k_tot, perc=1, num_of_threads=1)

        Estimate the probabilities of belonging to the classes for each target in x.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Target samples.

        f : lambda function, default=lambda k_tot, dim_plus_one: dim_plus_one - k_tot
            Function that compute polling weights of different classifiers

        perc : float, default=1
            Percentage of classifier to take into account for the prediction.

        num_of_threads : int, default=1
            Number of thread to use.

        Returns
        -------
        y : array of shape (n_queries, 2)
            Estimated probabilities of belonging to class 0 and class 1 respectively for each given target sample.

        """

        if len(self.classifiers) == 0:
            raise Exception("You have to train the classifier before predict classes.")

        self.classifiers.sort(key=lambda cl: cl.k[0] + cl.k[1])
        best_classifiers = self.classifiers[:round(len(self.classifiers) * perc)]

        if len(best_classifiers) == 0:
            raise Exception("perc is too low. No classifiers are selected.")

        pool = Pool(num_of_threads)
        predictions, weights = zip(*pool.starmap(self._predict_proba_single,
                                                 [(x, f, cl) for cl in best_classifiers]))  # multithreading predict

        np.asarray(weights)
        np.asarray(predictions)

        y = np.transpose(predictions).dot(weights)
        y = y / weights.sum()

        y = y.round(10)  # to avoid approximation problems (ex. probability >1)
        return np.append(1 - y, y, axis=1)  # {prob_for_class_0, prob_for_class_1}

    @staticmethod
    def _predict_proba_single(x, weight_function, classifier):
        k_tot = classifier.k[0] + classifier.k[1]
        n_plus_one = classifier.n[0] + classifier.n[1] + 1
        y = [classifier.predict(x), weight_function(k_tot, n_plus_one)]
        return y

    def predict(self, x, f=lambda k_tot, dim_plus_one: dim_plus_one - k_tot, perc=1, num_of_threads=1):
        """
        predict(self, x, f=lambda k_tot, dim_plus_one: dim_plus_one - k_tot, perc=1, num_of_threads=1)

        Predict class labels for the provided data.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Target samples.

        f : lambda function, default=lambda k_tot, dim_plus_one: dim_plus_one - k_tot
            Function that compute polling weights of different classifiers

        perc : float, default=1
            Percentage of classifier to take into account for the prediction.

        num_of_threads : int, default=1
            Number of thread to use.

        Returns
        -------
        y : array of shape (n_queries,)
            Predicted class labels for each data sample.

        """

        probas = self.predict_proba(x, f, perc, num_of_threads)
        return np.round(probas[:, 1]).astype(int)

    def score(self, x, y, f=lambda k_tot, dim_plus_one: dim_plus_one - k_tot, perc=1, num_of_threads=1):
        """
        score(self, x, y, f=lambda k_tot, dim_plus_one: dim_plus_one - k_tot, perc=1, num_of_threads=1)

        Return the accuracy score of the given test data and labels.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features)
            Test samples.

        y : array-like of shape (n_queries,)
            True class of x.

        f : lambda function, default=lambda k_tot, dim_plus_one: dim_plus_one - k_tot
            Function that compute polling weights of different classifiers

        perc : float, default=1
            Percentage of classifier to take into account for the prediction.

        num_of_threads : int, default=1
            Number of thread to use.

        Returns
        -------
        score : float
            The accuracy of the predicted labels of x with respect to y.

        """

        y = np.array(y)

        if len(x) != len(y):
            raise Exception("Features and labels must have the same length")

        y_predicted = self.predict(x, f, perc, num_of_threads)
        score = np.count_nonzero(y_predicted == y) / len(y)
        return score
