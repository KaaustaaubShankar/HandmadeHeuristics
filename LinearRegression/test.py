import numpy as np
from LinearRegression import LinearRegression

np.random.seed(42)
X = 2 * np.random.rand(100, 3)  # 100 samples, 3 features
true_weights = np.array([1,3,6])
y = X.dot(true_weights) + 12 + np.random.randn(100)  # Adding some noise

# Initialize and train the model
model = LinearRegression(learning_rate=0.01, loss_function='mse', features=3)
loss_history = model.train(X, y, epochs=1000)

# Predict on new data
predictions = model.predict(X[:5])  # Predict on first 5 samples
print("Predictions:", predictions)

model.printWeights()
