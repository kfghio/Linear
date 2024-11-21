import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('processed_manga_data.csv')

# print(data['status'].unique())
# [2 1 0 3]

# 2 (Ongoing) -> 1 (активный)
# Остальные классы -> 0 (не активный)
data['status'] = data['status'].replace({0: 0, 1: 0, 3: 0, 2: 1})

# print(data['status'].unique())
# [1 0]

data.to_csv('processed_manga_data_updated.csv', index=False)

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Протестим
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")