import numpy as np


class GEMBallsClassifier:

    def __init__(self, c0=1, c1=1):
        self.classifier = []
        self.c = [c0, c1]
        self.k = [0, 0]
        self.n = [0, 0]

    def fit(self, x, y, first=0):
        if any([label != 0 and label != 1 for label in y]):
            raise Exception("The training must contain 0 or 1 labelled points only")
        if len(y) != len(x):
            raise Exception("Features and labels must have the same length")

        self.classifier = []
        features_labels = [list(a) for a in zip(x, y)]
        current_x, current_y = features_labels.pop(first)

        self.n[0] = np.count_nonzero(np.array([f_l[1] for f_l in features_labels]) == 0)
        self.n[1] = np.count_nonzero(np.array([f_l[1] for f_l in features_labels]) == 1)
        self.k = [0, 0]

        while True:
            opposite_label = 1-current_y
            features_labels.sort(key=lambda f_l: np.linalg.norm(f_l[0] - current_x))
            opposite_positions = np.nonzero(np.array([f_l[1] for f_l in features_labels]) == opposite_label)[0]
            support_position = opposite_positions[:self.c[opposite_label]]
            if self.c[opposite_label] > len(support_position):
                self.classifier.append([current_x, float('Inf'), current_y])
                self.k[opposite_label] += len(support_position) + 1
                break
            else:
                edge_point, edge_label = features_labels[int(support_position[-1])]
                self.classifier.append([current_x, np.linalg.norm(edge_point - current_x), current_y])
                self.k[opposite_label] += len(support_position)
                current_x, current_y = edge_point, edge_label
                features_labels = features_labels[int(support_position[-1])+1:]

    def predict_proba(self, x):
        y = self.predict(x)
        y = y.reshape(-1, 1)
        return np.append(1 - y, y, axis=1)

    def predict(self, x):
        if not self.classifier:
            raise Exception("You have to train the model before use it")
        y = []
        for elem_x in x:
            for sphere in self.classifier:
                if np.linalg.norm(sphere[0]-elem_x) < sphere[1]:
                    y.append(sphere[2])
                    break

        if len(y) != len(x):
            raise Exception("Dataset contains null values")

        return np.array(y)

    def score(self, x, y):
        if any([label != 0 and label != 1 for label in y]):
            raise Exception("The training must contain 0 or 1 labelled points only")
        if len(x) != len(y):
            raise Exception("Features and labels must have the same length")
        y_predicted = self.predict(x)
        score = np.count_nonzero(np.array(y_predicted) == np.array(y)) / len(y)
        return score
