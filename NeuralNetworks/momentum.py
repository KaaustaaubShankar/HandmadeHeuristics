import numpy as np
import matplotlib.pyplot as plt
def train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True):
    x_train, x_test, y_train, y_test = [], [], [], []
    unique_digits = np.unique(y_data)

    for digit in unique_digits:
        digit_indices = np.where(y_data == digit)[0]
        split_point = int(len(digit_indices) * split_percentage)
        train_indices = digit_indices[:split_point]
        test_indices = digit_indices[split_point:]
        x_train.append(x_data[train_indices])
        y_train.append(y_data[train_indices])
        x_test.append(x_data[test_indices])
        y_test.append(y_data[test_indices])

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    if shuffle_train:
        shuffle_indices = np.random.permutation(len(x_train))
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

    return x_train.T, x_test.T, y_train.T, y_test.T

x_data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt')  
y_data = np.loadtxt('HW3_datafiles/MNISTnumLabels5000_balanced.txt')
x_train, x_test,y_train,y_test = train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True)

# Define activation functions and their derivatives
def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability improvement
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Encapsulated Mean Squared Error loss and its gradient
def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mse_gradient(predictions, targets):
    m = predictions.shape[1]
    return 2 * (predictions - targets) / m

def cross_entropy_loss(predictions, targets):
    # Small epsilon for stability
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    m = targets.shape[1]
    log_likelihood = -np.sum(targets * np.log(predictions))
    return log_likelihood / m

def cross_entropy_gradient(predictions, targets):
    m = targets.shape[1]
    return (predictions - targets) / m

# Dictionary to map activation names to functions
activation_functions = {
    'relu': (ReLU, ReLU_deriv),
    'softmax': (softmax, None)
}

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.W = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5
        self.activation, self.activation_deriv = activation_functions[activation]
    
    def forward(self, X):
        self.Z = self.W.dot(X) + self.b
        self.A = self.activation(self.Z) if self.activation else self.Z
        return self.A

def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
    return one_hot_Y.T

def forward_prop(X, network):
    activations = [X]
    for layer in network:
        X = layer.forward(X)
        activations.append(X)
    return activations

def backward_prop(Y, activations, network, loss_gradient):
    one_hot_Y = one_hot(Y, activations[-1].shape[0])
    dA = loss_gradient(activations[-1], one_hot_Y)

    gradients = []
    for i in reversed(range(len(network))):
        layer = network[i]
        dZ = dA * (layer.activation_deriv(layer.Z) if layer.activation_deriv else 1)
        dW = dZ.dot(activations[i].T)
        db = np.sum(dZ, axis=1, keepdims=True)
        gradients.append((dW, db))
        dA = layer.W.T.dot(dZ)
    
    return gradients[::-1]

# Update function with momentum
def update_params_with_momentum(network, gradients, velocities, alpha, beta):
    for i, (dW, db) in enumerate(gradients):
        # Update the velocity terms for weights and biases
        velocities[i]['W'] = beta * velocities[i]['W'] + (1 - beta) * dW
        velocities[i]['b'] = beta * velocities[i]['b'] + (1 - beta) * db

        # Update the weights and biases using the velocity
        network[i].W -= alpha * velocities[i]['W']
        network[i].b -= alpha * velocities[i]['b']

def compute_accuracy(predictions, Y):
    predicted_classes = np.argmax(predictions, axis=0)
    return np.mean(predicted_classes == Y) * 100

def train_with_momentum(network, X, Y, alpha, epochs, loss_function, loss_gradient, beta):
    m = X.shape[1]

    # Initialize velocities for each layer in the network
    velocities = [{'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)} for layer in network]

    for epoch in range(epochs):
        activations = forward_prop(X, network)
        gradients = backward_prop(Y, activations, network, loss_gradient)
        update_params_with_momentum(network, gradients, velocities, alpha, beta)

        if epoch % 100 == 0:
            loss = loss_function(activations[-1], one_hot(Y, activations[-1].shape[0]))
            accuracy = compute_accuracy(activations[-1], Y)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

# User-Specified Network Configuration
input_size = 784  # Example input size for MNIST
output_size = 10  # Number of classes

# Get user inputs for hidden layer sizes, learning rate, and momentum coefficient
hidden_layer_sizes = list(map(int, input("Enter hidden layer sizes separated by commas (e.g., 200,100,50): ").split(',')))
alpha = float(input("Enter the learning rate (e.g., 0.01): "))
beta = float(input("Enter the momentum coefficient (e.g., 0.9): "))
epochs = int(input("Enter the epochs: "))

# Build the network dynamically based on user specifications
layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
network = [
    Layer(layer_sizes[i], layer_sizes[i+1], activation='relu' if i < len(hidden_layer_sizes) else 'softmax')
    for i in range(len(layer_sizes) - 1)
]

# Train the network using Cross Entropy loss and momentum
train_with_momentum(network, x_train, y_train, alpha, epochs, mse_loss, mse_gradient, beta)

# Test the network
test_activations = forward_prop(x_test, network)
test_accuracy = compute_accuracy(test_activations[-1], y_test)
print(f"Test accuracy: {test_accuracy:.2f}%")

def display_prediction(network, x_test, y_test, num_images=5):
    indices = np.random.choice(x_test.shape[1], num_images, replace=False)
    for idx in indices:
        image = x_test[:, idx].reshape(28, 28)
        label = y_test[idx]
        prediction = np.argmax(forward_prop(x_test[:, idx].reshape(-1, 1), network)[-1], axis=0)[0]
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title(f"True Label: {label}, Predicted: {prediction}")
        plt.axis('off')
        plt.show()

# Display random predictions
display_prediction(network, x_test, y_test, num_images=5)
print()
# Display the output layer weights and biases
def display_output_layer(network):
    output_layer = network[-1]
    print("Output Layer Weights:\n", output_layer.W)
    print("Output Layer Biases:\n", output_layer.b)

display_output_layer(network)
