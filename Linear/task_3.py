import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from GradientDescentClassifier_2 import GradientDescentClassifier


class GradientDescentClassifierWrapper:
    def __init__(self, **params):
        self.model = GradientDescentClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"model": self.model}


data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(1500)

X = limited_data.drop('status', axis=1)
y = limited_data['status']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

gd_clf = GradientDescentClassifierWrapper(loss="logistic", l1_ratio=0.5, alpha=1.0, max_iter=1000)

train_sizes, train_scores, test_scores = learning_curve(
    gd_clf, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy'
)


train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(
    SVC(C=1, kernel='linear', max_iter=1000), X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy'
)


train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

train_scores_mean_svm = np.mean(train_scores_svm, axis=1)
test_scores_mean_svm = np.mean(test_scores_svm, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Train Accuracy (Logistic Regression)')
plt.plot(train_sizes, test_scores_mean, label='Test Accuracy (Logistic Regression)')

plt.plot(train_sizes_svm, train_scores_mean_svm, label='Train Accuracy (SVM)')
plt.plot(train_sizes_svm, test_scores_mean_svm, label='Test Accuracy (SVM)')

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curves for Logistic Regression and SVM")
plt.legend()
plt.show()
