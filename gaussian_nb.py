import numpy as np

class GaussianNB:
    def __init__(self):
        self.class_stats = {}   # mean & variance per feature per class
        self.class_priors = {}  # prior probabilities

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes model"""
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]  # data belonging to class c
            self.class_stats[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0)
            }
            self.class_priors[c] = len(X_c) / len(X)

    def _gaussian_prob(self, x, mean, var):
        """Gaussian Probability Density Function"""
        eps = 1e-6  # avoid division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
        exponent = np.exp(-(x - mean) ** 2 / (2 * (var + eps)))
        return coeff * exponent

    def predict(self, X):
        """Predict class labels for X"""
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.class_priors[c])   # log to avoid underflow
                likelihood = np.sum(
                    np.log(self._gaussian_prob(x, self.class_stats[c]["mean"], self.class_stats[c]["var"]))
                )
                posteriors[c] = prior + likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

