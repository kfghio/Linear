import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from SVM_3 import SVM

kernels = ["linear", "polynomial", "rbf"]
Cs = [0.1, 1, 10]
max_iters = [10]
tols = [1e-4, 1e-3]

data = pd.read_csv('processed_manga_data_updated.csv')
limited_data = data.head(5713)
X = limited_data.drop('status', axis=1)
y = limited_data['status']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


y_train = np.where(y_train == y_train.unique()[0], -1, 1)
y_test = np.where(y_test == y_test.unique()[0], -1, 1)

best_params = None
best_score = 0

for kernel in kernels:
    for C in Cs:
        for tol in tols:
            for max_iter in max_iters:
                print(f"Тестируем: kernel={kernel}, C={C}, tol={tol}, max_iter={max_iter}")

                model = SVM(kernel=kernel, C=C, tol=tol, max_iter=max_iter)

                model.fit(X_train, y_train)

                y_val_pred = model.predict(X_test)

                score = accuracy_score(y_test, y_val_pred)
                print(f"Точность: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_params = {"kernel": kernel, "C": C, "tol": tol, "max_iter": max_iter}

print("Лучшие параметры:", best_params)
print(f"Лучшая точность: {best_score:.2f}")
