from itertools import product

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from GradientDescentClassifier_2 import GradientDescentClassifier

losses = ["logistic", "exponential", "hinge"]
l1_ratios = [0.1, 0.5, 0.9]
alphas = [0.01, 0.1, 1]
learning_rates = [0.001, 0.01, 0.1]
max_iters = [10]


best_params = None
best_accuracy = 0

data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(5713)
X = limited_data.drop('status', axis=1)
y = limited_data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gd_clf = GradientDescentClassifier(loss="logistic", l1_ratio=0.5, alpha=1.0)
gd_clf.fit(X_train, y_train)
y_pred = gd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

for loss, l1_ratio, alpha, lr, max_iter in product(losses, l1_ratios, alphas, learning_rates, max_iters):
    gd_clf = GradientDescentClassifier(
        loss=loss, l1_ratio=l1_ratio, alpha=alpha, learning_rate=lr, max_iter=max_iter
    )
    gd_clf.fit(X_train.to_numpy(), y_train.to_numpy())  # Преобразуем в numpy массивы
    y_pred = gd_clf.predict(X_test.to_numpy())
    acc = accuracy_score(y_test, y_pred)

    if acc > best_accuracy:
        best_accuracy = acc
        best_params = (loss, l1_ratio, alpha, lr, max_iter)

print("Лучшие параметры:", best_params)
print(f"Лучшая точность: {best_accuracy:.2f}")
