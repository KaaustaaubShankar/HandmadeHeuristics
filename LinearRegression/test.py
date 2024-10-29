import numpy as np
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
def r2_score(y_pred,y_true):
    y_mean = np.mean(y_true)
    SSR = np.sum((y_true - y_pred)**2)
    SST = np.sum((y_true - y_mean)**2)
    return 1 - (SSR / SST)


np.random.seed(42)
X = 2 * np.random.rand(100, 3)  # 100 samples, 3 features
true_weights = np.array([1,3,6])
y = X.dot(true_weights) + 12 + np.random.randn(100)  # Adding some noise

# Initialize and train the model
model = LinearRegression(learning_rate=0.01, loss_function='mse', features=3)
loss_history = model.train(X, y, epochs=1000)

# Predict on new data
predictions = model.predict(X)  # Predict on first 5 samples

r_squared = r2_score(y, predictions)
print(f"R² Score: {r_squared:.4f}")


plt.figure(figsize=(8, 6))

# Scatter plot for True vs. Predicted values
plt.scatter(y, predictions, color="dodgerblue", label="Predicted Values", alpha=0.6, edgecolor="k")

# Best fit line (regression line)
m, b = np.polyfit(y, predictions, 1)
plt.plot(y, m * y + b, color="green", linestyle="--", label="Best Fit Line")

# Perfect prediction line (45-degree line)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Perfect Prediction Line")

# Labels and title
plt.xlabel("True Values (y)")
plt.ylabel("Predicted Values (y)")
plt.title(f"True vs. Predicted Values with R² = {r_squared:.4f}")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.show()