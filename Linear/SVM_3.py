import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, kernel="linear", C=1.0, tol=1e-4, max_iter=10):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0

    def _kernel_function(self, x1, x2):
        if self.kernel == "linear":
            return np.dot(x1, x2)
        elif self.kernel == "polynomial":
            degree = 3
            return (1 + np.dot(x1, x2)) ** degree
        elif self.kernel == "rbf":
            gamma = 0.5
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def _compute_kernel_matrix(self, X):
        if self.kernel == "linear":
            return np.dot(X, X.T)
        elif self.kernel == "rbf":
            gamma = 0.5
            squared_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
            return np.exp(-gamma * (squared_norms - 2 * np.dot(X, X.T) + squared_norms.T))
        else:
            n = X.shape[0]
            return np.array([[self._kernel_function(X[i], X[j]) for j in range(n)] for i in range(n)])

    def fit(self, X, y):
        self.X = X
        self.y = y
        n, m = self.X.shape
        self.alphas = np.zeros(n)
        self.b = 0

        K = self._compute_kernel_matrix(self.X)
        c1 = 0
        for _ in range(self.max_iter):
            c = 0
            c1 = c1 + 1
            alpha_prev = self.alphas.copy()

            for i in range(n):
                c = c + 1
                print(c, "///", c1)
                E_i = self._decision_function(self.X[i]) - self.y[i]

                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (
                        self.y[i] * E_i > self.tol and self.alphas[i] > 0):
                    j = np.random.choice([k for k in range(n) if k != i])
                    E_j = self._decision_function(self.X[j]) - self.y[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    if self.y[i] != self.y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_j_old + alpha_i_old - self.C)
                        H = min(self.C, alpha_j_old + alpha_i_old)

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    b1 = self.b - E_i - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - self.y[j] * (
                                self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - self.y[j] * (
                                self.alphas[j] - alpha_j_old) * K[j, j]

                    self.b = (b1 + b2) / 2 if 0 < self.alphas[i] < self.C else b1

            if np.allclose(self.alphas, alpha_prev):
                break

    def _decision_function(self, x):
        return sum(
            self.alphas[i] * self.y[i] * self._kernel_function(x, self.X[i]) for i in range(len(self.alphas))) + self.b

    def predict(self, X):
        return np.sign([self._decision_function(x) for x in X])


data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(100)
X = limited_data.drop('status', axis=1)
y = limited_data['status']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


y_train = np.where(y_train == y_train.unique()[0], -1, 1)
y_test = np.where(y_test == y_test.unique()[0], -1, 1)


svm = SVM(kernel="linear", C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
