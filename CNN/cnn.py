import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Utility Functions
def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtract max for stability
    return exps / np.sum(exps, axis=-1, keepdims=True)

def cross_entropy_loss(predicted, actual):
    return -np.mean(np.sum(actual * np.log(predicted + 1e-15), axis=-1))

def cross_entropy_grad(predicted, actual):
    return predicted - actual

# Base Layer Class
class Layer:
    def forward(self, input):
        raise NotImplementedError()

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError()

# Conv2D Layer
class Conv2D(Layer):
    def __init__(self, kernel_size, num_kernels, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(num_kernels, kernel_size, kernel_size) * 0.1
        self.biases = np.random.randn(num_kernels) * 0.1

    def forward(self, input):
        self.input = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        self.input_h, self.input_w = self.input.shape
        self.output_h = (self.input_h - self.kernel_size) // self.stride + 1
        self.output_w = (self.input_w - self.kernel_size) // self.stride + 1
        self.output = np.zeros((self.num_kernels, self.output_h, self.output_w))

        for k in range(self.num_kernels):
            for i in range(0, self.output_h):
                for j in range(0, self.output_w):
                    region = self.input[i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                    self.output[k, i, j] = np.sum(region * self.kernels[k]) + self.biases[k]

        return self.output

    def backward(self, grad_output, learning_rate=0.01):
        grad_input = np.zeros_like(self.input)
        grad_kernels = np.zeros_like(self.kernels)
        grad_biases = np.zeros_like(self.biases)

        for k in range(self.num_kernels):
            for i in range(0, self.output_h):
                for j in range(0, self.output_w):
                    region = self.input[i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                    grad_kernels[k] += grad_output[k, i, j] * region
                    grad_biases[k] += grad_output[k, i, j]
                    grad_input[i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += grad_output[k, i, j] * self.kernels[k]

        # Update weights
        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

# ReLU Layer
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate=0.01):
        return grad_output * (self.input > 0)

# MaxPooling Layer
class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        self.num_channels, self.input_h, self.input_w = input.shape
        self.output_h = (self.input_h - self.pool_size) // self.stride + 1
        self.output_w = (self.input_w - self.pool_size) // self.stride + 1
        self.output = np.zeros((self.num_channels, self.output_h, self.output_w))

        for c in range(self.num_channels):
            for i in range(0, self.output_h):
                for j in range(0, self.output_w):
                    region = input[c, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                    self.output[c, i, j] = np.max(region)

        return self.output

    def backward(self, grad_output, learning_rate=0.01):
        grad_input = np.zeros_like(self.input)

        for c in range(self.num_channels):
            for i in range(0, self.output_h):
                for j in range(0, self.output_w):
                    region = self.input[c, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                    max_value = np.max(region)
                    grad_region = (region == max_value) * grad_output[c, i, j]
                    grad_input[c, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size] += grad_region

        return grad_input

# Flatten Layer
class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()

    def backward(self, grad_output, learning_rate=0.01):
        return grad_output.reshape(self.input_shape)

# FullyConnected Layer
class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.random.randn(output_size) * 0.1

    def forward(self, input):
        self.input = input
        input = input.flatten()
        self.output = np.dot(self.weights, input) + self.biases
        return self.output

    def backward(self, grad_output, learning_rate=0.01):
        # Update weights for each sample in the batch
        grad_input = np.zeros_like(self.input)
        for i in range(grad_output.shape[0]):
            grad_weights = np.outer(grad_output[i], self.input)
            grad_biases = grad_output[i]
            grad_input += np.dot(self.weights.T, grad_output[i])

            # Update weights
            self.weights -= learning_rate * grad_weights
            self.biases -= learning_rate * grad_biases

        return grad_input

# Softmax Layer
class Softmax(Layer):
    def forward(self, input):
        exps = np.exp(input - np.max(input))  # Subtract max for numerical stability
        self.output = exps / np.sum(exps)
        return self.output

    def backward(self, grad_output, learning_rate=0.01):
        return grad_output

# Network Class
class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad_output, learning_rate=0.01):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

# Training Function
def train(network, x_train, y_train, epochs, batch_size, learning_rate):
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            outputs = []
            for x in x_batch:
                outputs.append(network.forward(x))
            outputs = np.array(outputs)

            # Compute loss
            loss = cross_entropy_loss(outputs, y_batch)

            # Backward pass
            grad_loss = cross_entropy_grad(outputs, y_batch)
            network.backward(grad_loss, learning_rate)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess Data
x_train = x_train[:1000] / 255.0  # Normalize to range [0, 1]
y_train = to_categorical(y_train[:1000], num_classes=10)  # One-hot encode labels

# Initialize the Network
net = Network()
net.add_layer(Conv2D(kernel_size=3, num_kernels=8, stride=1, padding=1))
net.add_layer(ReLU())
net.add_layer(MaxPooling(pool_size=2, stride=2))
net.add_layer(Flatten())
net.add_layer(FullyConnected(input_size=8*14*14, output_size=10))  # MNIST 28x28 -> Conv2D -> 14x14
net.add_layer(Softmax())

# Train the Network
train(net, x_train, y_train, epochs=5, batch_size=32, learning_rate=0.01)
