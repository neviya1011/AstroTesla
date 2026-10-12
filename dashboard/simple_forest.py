"""Small random-forest regressor implemented with NumPy only."""

import numpy as np


class SimpleRandomForestRegressor:
    """A simple bagged regression forest for environments without scikit-learn."""

    def __init__(self, n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must have one value per row")
        if len(X) < 2 * self.min_samples_leaf:
            raise ValueError("not enough rows to build a forest")

        self.n_features_in_ = X.shape[1]
        feature_count = max(1, int(np.sqrt(self.n_features_in_)))
        self.trees_ = []
        importance = np.zeros(self.n_features_in_, dtype=float)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            sample_indices = rng.integers(0, len(X), size=len(X))
            tree_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            tree_importance = np.zeros(self.n_features_in_, dtype=float)
            tree = self._build_tree(
                X[sample_indices], y[sample_indices], 0, feature_count, tree_rng, tree_importance
            )
            self.trees_.append(tree)
            importance += tree_importance

        total = importance.sum()
        self.feature_importances_ = importance / total if total else importance
        return self

    def predict(self, X):
        if not self.trees_:
            raise ValueError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = np.array([self._predict_tree(tree, X) for tree in self.trees_])
        return predictions.mean(axis=0)

    def _build_tree(self, X, y, depth, feature_count, rng, importance):
        value = float(np.mean(y))
        node = {"value": value}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.all(y == y[0]):
            return node

        parent_error = self._squared_error(y)
        best = None
        features = rng.choice(self.n_features_in_, size=feature_count, replace=False)
        for feature in features:
            order = np.argsort(X[:, feature], kind="mergesort")
            values = X[order, feature]
            sorted_y = y[order]
            split_positions = np.flatnonzero(values[1:] > values[:-1]) + 1
            if len(split_positions) > 32:
                choices = np.linspace(0, len(split_positions) - 1, 32, dtype=int)
                split_positions = split_positions[choices]

            for position in split_positions:
                if position < self.min_samples_leaf or len(y) - position < self.min_samples_leaf:
                    continue
                left_y = sorted_y[:position]
                right_y = sorted_y[position:]
                error = self._squared_error(left_y) + self._squared_error(right_y)
                if best is None or error < best[0]:
                    best = (error, int(feature), float((values[position - 1] + values[position]) / 2))

        if best is None or best[0] >= parent_error:
            return node

        error, feature, threshold = best
        left = X[:, feature] <= threshold
        right = ~left
        node.update({
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left], y[left], depth + 1, feature_count, rng, importance),
            "right": self._build_tree(X[right], y[right], depth + 1, feature_count, rng, importance),
        })
        importance[feature] += max(parent_error - error, 0.0)
        return node

    @staticmethod
    def _squared_error(values):
        if len(values) == 0:
            return 0.0
        centered = values - np.mean(values)
        return float(np.dot(centered, centered))

    @classmethod
    def _predict_tree(cls, tree, X):
        predictions = np.empty(len(X), dtype=float)
        for row_number, row in enumerate(X):
            node = tree
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            predictions[row_number] = node["value"]
        return predictions


class NumpyRidgeRegressor:
    """Small standardized ridge regressor."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        scaled = (X - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(X)), scaled])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scaled = (X - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(X)), scaled]) @ self.coef_


class MeanDaysRegressor:
    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=float)


class StraightLineRegressor:
    def __init__(self, mode, value, cap=120):
        self.mode = mode
        self.value = value
        self.cap = cap

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        rate = X[:, 3]
        target_level = self.value if self.mode == "absolute" else X[:, 10] - self.value
        remaining = X[:, 0] - target_level
        prediction = np.full(len(X), self.cap, dtype=float)
        moving = rate > 0
        prediction[moving] = remaining[moving] / rate[moving]
        return np.clip(prediction, 0, self.cap)


class SmoothedRateRegressor:
    def __init__(self, mode, value, cap=120):
        self.mode = mode
        self.value = value
        self.cap = cap

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        rate = -X[:, 5]
        target_level = self.value if self.mode == "absolute" else X[:, 10] - self.value
        remaining = X[:, 0] - target_level
        prediction = np.full(len(X), self.cap, dtype=float)
        moving = rate > 0
        prediction[moving] = remaining[moving] / rate[moving]
        return np.clip(prediction, 0, self.cap)
