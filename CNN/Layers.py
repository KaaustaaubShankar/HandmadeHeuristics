import numpy as np
import matplotlib.pyplot as plt

# Utilities
def train_test_split(x_data, y_data, split_percentage=0.8, shuffle=True):
    unique_classes = np.unique(y_data)
    x_train, x_test, y_train, y_test = [], [], [], []

    for cls in unique_classes:
        cls_indices = np.where(y_data == cls)[0]
        split_idx = int(len(cls_indices) * split_percentage)
        x_train.append(x_data[cls_indices[:split_idx]])
        y_train.append(y_data[cls_indices[:split_idx]])
        x_test.append(x_data[cls_indices[split_idx:]])
        y_test.append(y_data[cls_indices[split_idx:]])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    if shuffle:
        perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]

    return x_train.T, x_test.T, y_train.T, y_test.T

# Activation Functions
def relu(Z): return np.maximum(0, Z)
def relu_deriv(Z): return Z > 0
def softmax(Z): 
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def sigmoid_deriv(Z): return sigmoid(Z) * (1 - sigmoid(Z))

# Loss Functions
def mse_loss(pred, target): return np.mean((pred - target) ** 2)
def mse_grad(pred, target): return 2 * (pred - target) / pred.shape[1]

# One-Hot Encoding
def one_hot(Y, num_classes):
    one_hot_matrix = np.zeros((num_classes, Y.size))
    one_hot_matrix[Y.astype(int), np.arange(Y.size)] = 1
    return one_hot_matrix

# Layer Class
class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((output_size, 1))
        activations = {'relu': (relu, relu_deriv), 'softmax': (softmax, None), 'sigmoid': (sigmoid, sigmoid_deriv)}
        self.activate, self.activate_deriv = activations[activation]
    
    def forward(self, X):
        self.Z = self.W @ X + self.b
        self.A = self.activate(self.Z) if self.activate else self.Z
        return self.A

# Neural Network Operations
def forward_pass(X, layers):
    activations = [X]
    for layer in layers:
        X = layer.forward(X)
        activations.append(X)
    return activations

def backward_pass(Y, activations, layers, loss_grad, is_classification):
    dA = loss_grad(activations[-1], one_hot(Y, activations[-1].shape[0]) if is_classification else Y)
    grads = []

    for i in reversed(range(len(layers))):
        layer = layers[i]
        dZ = dA * (layer.activate_deriv(layer.Z) if layer.activate_deriv else 1)
        dW = dZ @ activations[i].T
        db = np.sum(dZ, axis=1, keepdims=True)
        grads.insert(0, (dW, db))
        dA = layer.W.T @ dZ
    
    return grads

def update_params(layers, grads, lr):
    for layer, (dW, db) in zip(layers, grads):
        layer.W -= lr * dW
        layer.b -= lr * db

# Training Function
def train_network(layers, X, Y, lr, epochs, batch_size, loss_fn, loss_grad, is_classification=True, verbose=False):
    m = X.shape[1]
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        X_shuffled, Y_shuffled = X.iloc[perm], Y.iloc[perm]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            activations = forward_pass(X_batch, layers)
            grads = backward_pass(Y_batch, activations, layers, loss_grad, is_classification)
            update_params(layers, grads, lr)
        
        if verbose and epoch % 10 == 0:
            predictions = forward_pass(X, layers)[-1]
            accuracy = np.mean(np.argmax(predictions, axis=0) == Y) * 100
            print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")

# Helper for Metrics
def compute_accuracy(predictions, Y):
    predicted_classes = np.argmax(predictions, axis=0)
    return np.mean(predicted_classes == Y) * 100

def create_confusion_matrix(predictions, Y, num_classes=10):
    predicted_classes = np.argmax(predictions, axis=0)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(Y, predicted_classes):
        matrix[true][pred] += 1
    return matrix

def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
    plt.show()
