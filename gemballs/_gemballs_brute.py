import numpy as np
from gemballs.gemballsclassifier import GEMBallsClassifier, Ball


class _LabelledPoint:

    def __init__(self, point, label):
        self.point = np.array(point)
        self.label = label


class _GEMBallsBrute(GEMBallsClassifier):

    algorithm_name = 'brute'

    def fit(self, x, y, first=0):
        self.classifier = []
        self.k = [0, 0]
        self.n = [0, 0]
        dataset = []

        for point, label in zip(x, y):
            self.n[label] += 1
            dataset.append(_LabelledPoint(point, label))

        current_point = dataset.pop(first)
        self.n[current_point.label] -= 1

        while True:
            opposite_label = 1-current_point.label
            dataset.sort(key=lambda labelled_point: np.linalg.norm(labelled_point.point - current_point.point))
            opposite_positions = [index for index, labelled_point in enumerate(dataset)
                                  if labelled_point.label == opposite_label]
            support_positions = opposite_positions[:self.c[opposite_label]]

            if self.c[opposite_label] > len(support_positions):
                sphere = Ball(current_point.point, float('Inf'), current_point.label)
                self.classifier.append(sphere)
                self.k[opposite_label] += len(support_positions) + 1
                break

            else:
                edge_point = dataset[support_positions[-1]]
                radius = np.linalg.norm(edge_point.point - current_point.point)
                sphere = Ball(current_point.point, radius, current_point.label)
                self.classifier.append(sphere)
                self.k[opposite_label] += len(support_positions)
                current_point = edge_point
                dataset = dataset[support_positions[-1]+1:]
