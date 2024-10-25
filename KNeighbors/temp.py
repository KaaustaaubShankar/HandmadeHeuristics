import matplotlib.pyplot as plt
import numpy as np
from KNN import KNearestNeighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(iris.target_names[y])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

k = 5
knn = KNearestNeighbors(k=k)
knn.fit(X_train, y_train)

test_point = X_test[0]
true_label = label_encoder.inverse_transform([y_test[0]])[0]

predicted_label = label_encoder.inverse_transform([knn.predict([test_point])[0]])[0]

plt.figure(figsize=(10, 6))
added_classes = set()
for i, (x, y_val) in enumerate(X_train):
    color = ['red', 'green', 'blue'][y_train[i]]
    label = f'Class {label_encoder.inverse_transform([y_train[i]])[0]}' if y_train[i] not in added_classes else ""
    plt.scatter(x, y_val, color=color, label=label)
    added_classes.add(y_train[i])

plt.scatter(test_point[0], test_point[1], color='yellow', edgecolor='black', s=100, label='Test Point')

distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
nearest_indices = np.argsort(distances)[:k]

for index in nearest_indices:
    plt.scatter(X_train[index, 0], X_train[index, 1], facecolors='none', edgecolor='cyan', s=200, linewidths=2)

for index in nearest_indices:
    plt.plot([test_point[0], X_train[index, 0]], [test_point[1], X_train[index, 1]], 'k--', lw=1)

plt.legend()
plt.title(f'KNN Visualization (True Label: {true_label}, Predicted: {predicted_label})')
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.grid(True)
plt.show()
