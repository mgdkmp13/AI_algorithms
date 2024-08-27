import copy

import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_help(self, y):
        n = len(y)
        if n == 0:
            return 0, 0

        ones_num = np.count_nonzero(y)
        zeros_num = n - ones_num

        return ones_num, zeros_num

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        # TODO find position of best data split
        for split in possible_splits:
            left_child = y[:split+1]
            right_child = y[split+1:]
            l_pos, l_neg = self.gini_help(left_child)
            r_pos, r_neg = self.gini_help(right_child)

            left = l_pos + l_neg
            right = r_pos + r_neg

            if left == 0 or right == 0:
                continue

            gini_left = 1 - (l_pos / left) ** 2 - (l_neg / left) ** 2
            gini_right = 1 - (r_pos / right) ** 2 - (r_neg / right) ** 2

            gini_gain = 1 - (left * gini_left) / (left + right) - (right * gini_right) / (left + right)

            if gini_gain > best_gain:
                best_gain = gini_gain
                best_idx = split

        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None

        # TODO implement feature selection
        if feature_subset is not None:
            selected_features = np.random.choice(X.shape[1], size=feature_subset, replace=False)
        else:
            selected_features = range(X.shape[1])

        for d in selected_features:
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
