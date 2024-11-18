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

# region activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability improvement
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(Z):
    s = sigmoid(Z)
    return s * (1 - s)

# Encapsulated Mean Squared Error loss and its gradient
def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mse_gradient(predictions, targets):
    m = predictions.shape[1]
    return 2 * (predictions - targets) / m


#endregion
class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        print(activation)
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.b = np.zeros((output_size, 1))
        self.activationFunctions = {
                'relu': (ReLU, ReLU_deriv),
                'softmax': (softmax, None),
                'sigmoid': (sigmoid, sigmoid_deriv)
            }
        self.activation, self.activation_deriv = self.activationFunctions[activation]
    
    def forward(self, X):
        self.Z = self.W.dot(X) + self.b
        self.A = self.activation(self.Z) if self.activation else self.Z
        return self.A
#region neural network related code

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

def backward_prop(Y, activations, network, loss_gradient, isClassification):
    if isClassification:
        one_hot_Y = one_hot(Y, activations[-1].shape[0])
        dA = loss_gradient(activations[-1], one_hot_Y)
    else:
        # For reconstruction error, Y is already in the correct format
        dA = loss_gradient(activations[-1], Y)

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

def train_with_momentum(network, X, Y, alpha, epochs, loss_function, loss_gradient, beta, 
                        batch_size=128, verbose=False, X_test=None, Y_test=None, 
                        isClassification=True, target_error=None):
    m = X.shape[1]
    velocities = [{'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)} for layer in network]
    trainErrorFractions, testErrorFractions = [], []
    
    for epoch in range(epochs):
        batch_indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X[:, batch_indices]
        if isClassification:
            Y_batch = Y[batch_indices]
        else:
            Y_batch = Y[:, batch_indices]
        
        # Forward and backward propagation
        activations = forward_prop(X_batch, network)
        gradients = backward_prop(Y_batch, activations, network, loss_gradient, isClassification)
        update_params_with_momentum(network, gradients, velocities, alpha, beta)

        #metrics every 10 epochs
        if verbose and epoch % 10 == 0 or target_error is not None:
            # Calculate training metrics
            train_activations = forward_prop(X, network)
            if isClassification:
                train_loss = loss_function(train_activations[-1], one_hot(Y, train_activations[-1].shape[0]))
                train_error_fraction = 1 - compute_accuracy(train_activations[-1], Y) / 100
            else:
                train_loss = loss_function(train_activations[-1], Y)
                train_error_fraction = train_loss
            
            # Calculate test metrics
            test_error_fraction = None
            if X_test is not None and Y_test is not None:
                test_activations = forward_prop(X_test, network)
                if isClassification:
                    test_loss = loss_function(test_activations[-1], one_hot(Y_test, test_activations[-1].shape[0]))
                    test_error_fraction = 1 - compute_accuracy(test_activations[-1], Y_test) / 100
                else:
                    test_loss = loss_function(test_activations[-1], Y_test)
                    test_error_fraction = test_loss

            # Append both errors together
            trainErrorFractions.append(train_error_fraction)
            if test_error_fraction is not None:
                testErrorFractions.append(test_error_fraction)
            
            if verbose:
                print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Error Fraction = {train_error_fraction}")
                if test_error_fraction is not None:
                    print(f"Epoch {epoch}: Test Loss = {test_loss}, Test Error Fraction = {test_error_fraction}")

            # Check for target error condition
            if target_error is not None and train_error_fraction <= target_error:
                print(f"Target error reached: {train_error_fraction} <= {target_error}")
                break

    return trainErrorFractions, testErrorFractions,epoch


def create_confusion_matrix(predictions, true_labels):

    predicted_classes = np.argmax(predictions, axis=0)
    true_labels = true_labels.astype(int)  # Convert labels to integers
    n_classes = 10
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(true_labels, predicted_classes):
        confusion_matrix[true_label][pred_label] += 1
        
    return confusion_matrix

def plot_confusion_matrix(train_confusion_matrix, test_confusion_matrix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot training confusion matrix
    im1 = ax1.imshow(train_confusion_matrix, cmap='YlOrRd')
    plt.colorbar(im1, ax=ax1)
    
    # Plot test confusion matrix
    im2 = ax2.imshow(test_confusion_matrix, cmap='YlOrRd')
    plt.colorbar(im2, ax=ax2)
    
    
    for ax, matrix, title in [(ax1, train_confusion_matrix, 'Training Set'),
                             (ax2, test_confusion_matrix, 'Test Set')]:
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix for {title}')
        
        
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, matrix[i, j],
                             ha="center", va="center",
                             color="black" if matrix[i, j] < matrix.max()/2 else "white")
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_error_fractions(train_errors, test_errors, save_interval=10):

    # Calculate actual epochs based on save_interval
    epochs = np.arange(0, len(train_errors))  # Modified this line
    
    # Create figure with specific size and DPI
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Plot lines with increased line width
    plt.plot(epochs, train_errors, 'b-', label='Training Error', linewidth=2, alpha=0.8)
    plt.plot(epochs, test_errors, 'r-', label='Test Error', linewidth=2, alpha=0.8)
    
    # Create boxes for start and end values
    box_style = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8)
    
    # Add annotations with boxes
    plt.annotate(f'Start: {train_errors[0]:.3f}', 
                xy=(epochs[0], train_errors[0]),
                xytext=(20, 20),
                textcoords='offset points',
                bbox=box_style,
                color='blue',
                fontsize=9)
    
    plt.annotate(f'End: {train_errors[-1]:.3f}',
                xy=(epochs[-1], train_errors[-1]),
                xytext=(-20, 20),
                textcoords='offset points',
                bbox=box_style,
                color='blue',
                fontsize=9,
                ha='right')
    
    plt.annotate(f'Start: {test_errors[0]:.3f}',
                xy=(epochs[0], test_errors[0]),
                xytext=(20, -30),
                textcoords='offset points',
                bbox=box_style,
                color='red',
                fontsize=9)
    
    plt.annotate(f'End: {test_errors[-1]:.3f}',
                xy=(epochs[-1], test_errors[-1]),
                xytext=(-20, -30),
                textcoords='offset points',
                bbox=box_style,
                color='red',
                fontsize=9,
                ha='right')
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error Fraction', fontsize=12)
    plt.title('Training and Test Error Over Time', fontsize=14, pad=20)
    
    # Add legend with final values
    plt.legend([
        f'Training Error (Final: {train_errors[-1]:.3f})',
        f'Test Error (Final: {test_errors[-1]:.3f})'
    ], loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()