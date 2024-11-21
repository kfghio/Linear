import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape
        I = np.eye(m)
        I[0, 0] = 0
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights


class LinearClassifierRidge:
    def __init__(self, alpha=1.0):
        self.model = RidgeRegression(alpha=alpha)

    def fit(self, X, y):
        y = np.where(y > 0, 1, -1)
        self.model.fit(X, y)

    def predict(self, X):
        return np.sign(self.model.predict(X))


data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(5713)
X = limited_data.drop('status', axis=1)
y = limited_data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_clf = LinearClassifierRidge(alpha=1.0)
ridge_clf.fit(X_train, y_train)
y_pred = ridge_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")