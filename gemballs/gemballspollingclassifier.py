from .gemballsclassifier_kdtree import GEMBallsKDTree
import numpy as np
from multiprocessing import Pool
import os


class GemballsPollingClassifier:

    def __init__(self, c0=1, c1=1, max_classifiers=0):
        self.classifiers = []
        self.c = [c0, c1]
        self.max_classifiers = max_classifiers

    def fit(self, x, y, num_of_threads=len(os.sched_getaffinity(0))):
        y = np.array(y)
        x = np.array(x)

        max_classifiers = len(x[:self.max_classifiers]) if self.max_classifiers != 0 else len(x)

        pool = Pool(num_of_threads)
        self.classifiers = pool.starmap(self.fit_single,
                                        [(x, y, i) for i in range(0, max_classifiers)])  # multithreading fit

    def fit_single(self, x, y, i):
        gemballs = GEMBallsKDTree(self.c[0], self.c[1])
        gemballs.fit(x, y, i)
        return gemballs

    def predict_proba(self, x, f=lambda ktot, dim_plus_one: 1 - ktot / dim_plus_one, perc=1,
                      num_of_threads=len(os.sched_getaffinity(0))):
        """Argument f is the function that computes weights given ktot and N+1
           perc is the percentage of the most promising classifiers to use. """
        if len(self.classifiers) == 0:
            raise Exception("You have to train the model before use fit.")

        self.classifiers.sort(key=lambda cl: cl.k[0] + cl.k[1])
        best_classifiers = self.classifiers[:round(len(self.classifiers) * perc)]

        if len(best_classifiers) == 0:
            raise Exception("perc is too low. No clasifiers are selected.")

        pool = Pool(num_of_threads)
        ys, weights = zip(*pool.starmap(self.predict_proba_single,
                                        [(x, f(cl.k[0] + cl.k[1], cl.n[0] + cl.n[1] + 1), cl) for cl in
                                         best_classifiers]))  # multithreading predict

        weights = np.array(weights)
        y = np.transpose(np.array(ys)).dot(weights)
        y = y / weights.sum()
        y = y.reshape(-1, 1)
        y = y.round(10)  # to avoid approximation problems (ex. probability >1)
        return np.append(1 - y, y, axis=1)  # len(x)x2 matrix, {prob_of_0_class, prob_of_1_class}

    @staticmethod
    def predict_proba_single(x, weight, classifier):
        y = [classifier.predict(x), weight]
        return y

    def predict(self, x, f=lambda ktot, dim_plus_one: 1 - ktot / dim_plus_one):
        probas = self.predict_proba(x, f)
        return np.round(probas[:, 1]).astype(int)

    def score(self, x, y, f=lambda ktot, dim_plus_one: 1 - ktot / dim_plus_one):
        if any([label != 0 and label != 1 for label in y]):
            raise Exception("y must contain 0 or 1 only")
        if len(x) != len(y):
            raise Exception("Features and labels must have the same length")
        y_predicted = self.predict(x, f)
        score = np.count_nonzero(y_predicted == np.array(y)) / len(y)
        return score

    def mean_square_error(self, x, y, f=lambda ktot, dim_plus_one: 1 - ktot / dim_plus_one):
        if any([label != 0 and label != 1 for label in y]):
            raise Exception("y must contain 0 or 1 only")
        if len(x) != len(y):
            raise Exception("Features and labels must have the same length")
        y_predicted = self.predict_proba(x, f)
        square_norm = np.linalg.norm(y_predicted[:, 1] - y) ** 2
        return square_norm / len(y)
