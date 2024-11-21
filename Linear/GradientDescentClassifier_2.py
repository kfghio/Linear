import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GradientDescentClassifier:
    def __init__(self, loss="logistic", l1_ratio=0.5, alpha=1.0, learning_rate=0.01, max_iter=10):
        self.loss = loss
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def _compute_loss(self, margins):
        if self.loss == "logistic":
            return np.log(1 + np.exp(-margins))
        elif self.loss == "exponential":
            return np.exp(-margins)
        elif self.loss == "hinge":
            return np.maximum(0, 1 - margins)

    def _compute_gradient(self, X, y, margins):
        if self.loss == "logistic":
            grad = -y / (1 + np.exp(margins))
        elif self.loss == "exponential":
            grad = -y * np.exp(-margins)
        elif self.loss == "hinge":
            grad = np.where(margins < 1, -y, 0)

        return grad @ X / len(y)

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = np.where(y > 0, 1, -1)
        n, m = X.shape
        self.weights = np.zeros(m)

        for _ in range(self.max_iter):
            margins = y * (X @ self.weights)
            grad = self._compute_gradient(X, y, margins)

            # Elastic Net регуляризация
            l1_grad = self.l1_ratio * np.sign(self.weights)
            l2_grad = (1 - self.l1_ratio) * self.weights
            regularization = self.alpha * (l1_grad + l2_grad)

            # Градиентный шаг
            self.weights -= self.learning_rate * (grad + regularization)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.sign(X @ self.weights)


data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(1500)


X = limited_data.drop('status', axis=1)
y = limited_data['status']


X = pd.get_dummies(X, drop_first=True)


y = y.factorize()[0]
y = np.where(y > 0, 1, -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

gd_clf = GradientDescentClassifier(loss="logistic", l1_ratio=0.5, alpha=1.0, learning_rate=0.01, max_iter=100)
gd_clf.fit(X_train, y_train)

y_pred = gd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
