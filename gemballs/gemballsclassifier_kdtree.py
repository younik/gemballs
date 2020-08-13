import numpy as np
from scipy.spatial import cKDTree
from .gemballsclassifier import GEMBallsClassifier


class GEMBallsKDTree(GEMBallsClassifier):

    def fit(self, x, y, first=0):
        if any([label != 0 and label != 1 for label in y]):
            raise Exception("The training must contain 0 or 1 labelled points only")
        if len(y) != len(x):
            raise Exception("Features and labels must have the same length")

        # creates data structures
        self.classifier = []  # clean, for multiple fit
        current_point, current_label = x[first], y[first]
        np.delete(x, first, 0)
        np.delete(y, first, 0)
        opposite_label = 1 - current_label

        points = [[], []]
        for point, label in zip(x, y):
            points[label].append(point)
            self.n[label] += 1

        tree = [cKDTree(points[0]), cKDTree(points[1])]
        deleted_point = set()

        while True:
            opp_near_points_distances = tree[opposite_label].query(current_point, len(tree[opposite_label].data))

            border_point = None
            radius = 0
            opposite_point_in_sphere = 0
            for distance, index in zip(opp_near_points_distances[0], opp_near_points_distances[1]):
                opp_near_point = tuple(tree[opposite_label].data[index])
                if opp_near_point not in deleted_point:
                    self.k[opposite_label] += 1
                    opposite_point_in_sphere += 1
                    deleted_point.add(opp_near_point)
                    if opposite_point_in_sphere == self.c[opposite_label]:
                        radius = distance
                        self.classifier.append([current_point, radius, current_label])
                        border_point = opp_near_point
                        break

            if not border_point:  # exit condition
                self.classifier.append([current_point, float('Inf'), current_label])
                self.k[opposite_label] += 1
                break

            indexes = tree[current_label].query_ball_point(current_point, radius)
            deleted_point.update([tuple(tree[current_label].data[i]) for i in indexes])

            current_point = np.array(border_point)
            current_label, opposite_label = opposite_label, current_label
