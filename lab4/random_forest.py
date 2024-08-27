from collections import defaultdict
import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, params):
        self.forest = []
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X,y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Random forest accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X):
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        forest_predictions = list(map(lambda x: sum(x)/len(x), zip(*tree_predictions)))
        return forest_predictions

    def bagging(self, X, y):
        # TODO implement bagging
        indexes = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_selected, y_selected = X[indexes], y[indexes]
        return X_selected, y_selected





























'''
    n_samples = X.shape[0]
    n_selected_samples = int(0.63 * n_samples)

    selected_indices = np.random.choice(n_samples, size=n_selected_samples, replace=False)
    X_selected = X[selected_indices]
    y_selected = y[selected_indices]

    remaining_indices = np.random.choice(n_samples, size=n_samples - n_selected_samples, replace=True)
    X_remaining = X[remaining_indices]
    y_remaining = y[remaining_indices]

    X_bagged = np.concatenate([X_selected, X_remaining], axis=0)
    y_bagged = np.concatenate([y_selected, y_remaining], axis=0)
    return X_bagged, y_bagged
'''