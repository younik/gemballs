import numpy as np
from scipy.spatial import cKDTree
from gemballs.gemballsclassifier import GEMBallsClassifier, Ball


class _GEMBallsKDTree(GEMBallsClassifier):
    """
    This class implements the fit method using k-d tree.
    It can be used by specifying parameter algorithm='kd_tree' during the construction of GEMBallsClassifier.
    """

    algorithm_name = 'kd_tree'

    @staticmethod
    def construct_trees(x, y):
        points = [[], []]
        for point, label in zip(x, y):
            points[label].append(point)

        trees = [cKDTree(points[0]), cKDTree(points[1])]
        return trees

    @staticmethod
    def tree_remove(tree, indices):
        return cKDTree(np.delete(tree.data, indices, axis=0))

    def fit(self, x, y, first=0):
        self.classifier = []
        self.k = [0, 0]
        x = np.array(x)
        y = np.array(y)
        current_point = x[first]
        current_label = y[first]
        x = np.delete(x, first, axis=0)
        y = np.delete(y, first, axis=0)
        trees = self.construct_trees(x, y)
        self.n = [len(trees[0].data), len(trees[1].data)]

        while True:
            opposite_label = 1 - current_label

            if self.c[opposite_label] > len(trees[opposite_label].data):
                sphere = Ball(current_point, float('Inf'), current_label)
                self.classifier.append(sphere)
                self.k[opposite_label] += len(trees[opposite_label].data) + 1
                break

            else:
                support_distances, support_indices = trees[opposite_label].query(current_point, self.c[opposite_label])
                if type(support_distances) is not np.ndarray:
                    support_distances = [support_distances]
                    support_indices = [support_indices]
                edge_point = trees[opposite_label].data[support_indices[-1]]
                sphere = Ball(current_point, support_distances[-1], current_label)
                self.classifier.append(sphere)
                self.k[opposite_label] += len(support_indices)
                in_sphere_indices = trees[current_label].query_ball_point(sphere.center, sphere.radius)
                trees[current_label] = self.tree_remove(trees[current_label], in_sphere_indices)
                trees[opposite_label] = self.tree_remove(trees[opposite_label], support_indices)
                current_point = edge_point
                current_label = opposite_label
