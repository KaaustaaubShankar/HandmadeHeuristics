from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Layers import Layer, train_network, forward_pass, compute_accuracy, create_confusion_matrix, plot_confusion_matrix
from Layers import mse_loss, mse_grad
import numpy as np

# Fetch MNIST Dataset
def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X, Y = mnist.data / 255.0, mnist.target.astype(int)
    return X, Y

# Prepare the data
X, Y = load_mnist()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).T
X_test = scaler.transform(X_test).T

# Parameters
input_size = X_train.shape[0]
hidden_size = 128
output_size = 10
epochs = 50
batch_size = 128
learning_rate = 0.01

# Create Network
layers = [
    Layer(input_size, hidden_size, activation='relu'),
    Layer(hidden_size, output_size, activation='softmax')
]

# Train the network
train_network(
    layers,
    X_train,
    Y_train,
    lr=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    loss_fn=mse_loss,
    loss_grad=mse_grad,
    is_classification=True,
    verbose=True
)

# Evaluate the network
train_predictions = forward_pass(X_train, layers)[-1]
test_predictions = forward_pass(X_test, layers)[-1]

train_accuracy = compute_accuracy(train_predictions, Y_train)
test_accuracy = compute_accuracy(test_predictions, Y_test)

print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Confusion Matrices
train_conf_matrix = create_confusion_matrix(train_predictions, Y_train)
test_conf_matrix = create_confusion_matrix(test_predictions, Y_test)

# Plot Confusion Matrices
plot_confusion_matrix(train_conf_matrix, title="Training Confusion Matrix")
plot_confusion_matrix(test_conf_matrix, title="Test Confusion Matrix")
