import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(1500)

X = limited_data.drop('status', axis=1)
y = limited_data['status']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)

train_sizes, train_scores, test_scores = learning_curve(
    log_reg, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
)

svm_clf = SVC(C=1, kernel='linear', max_iter=1000)
train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(
    svm_clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
)

lin_reg = LogisticRegression(penalty=None, max_iter=1000)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

train_scores_mean_svm = np.mean(train_scores_svm, axis=1)
train_scores_std_svm = np.std(train_scores_svm, axis=1)
test_scores_mean_svm = np.mean(test_scores_svm, axis=1)
test_scores_std_svm = np.std(test_scores_svm, axis=1)

plt.figure(figsize=(10, 6))

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label="Train Accuracy (Logistic Regression)", alpha=1)
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label="Test Accuracy (Logistic Regression)", alpha=1)

plt.fill_between(train_sizes_svm, train_scores_mean_svm - train_scores_std_svm, train_scores_mean_svm + train_scores_std_svm, alpha=0.1, color='orange')
plt.fill_between(train_sizes_svm, test_scores_mean_svm - test_scores_std_svm, test_scores_mean_svm + test_scores_std_svm, alpha=0.1, color='red')
plt.plot(train_sizes_svm, train_scores_mean_svm, 'o-', color='orange', label="Train Accuracy (SVM)", alpha=0.5)
plt.plot(train_sizes_svm, test_scores_mean_svm, 'o-', color='red', label="Test Accuracy (SVM)", alpha=1)

plt.axhline(y=1 - mse, color='purple', linestyle='--', label=f"Linear Regression (1 - MSE)")

plt.xlabel("Training Size")
plt.ylabel("Accuracy / Error")
plt.title("Learning Curves with Confidence Intervals")
plt.legend()
plt.grid()
plt.show()
