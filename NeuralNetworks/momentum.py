import numpy as np
import matplotlib.pyplot as plt
from Layers import Layer
from Layers import train_test_split, ReLU, ReLU_deriv, softmax, mse_loss, mse_gradient
from Layers import forward_prop, train_with_momentum, compute_accuracy

x_data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt')  
y_data = np.loadtxt('HW3_datafiles/MNISTnumLabels5000_balanced.txt')
x_train, x_test,y_train,y_test = train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True)



# User-Specified Network Configuration
input_size = 784  # Example input size for MNIST
output_size = 10  # Number of classes

# Get user inputs for hidden layer sizes, learning rate, and momentum coefficient
def get_user_input(prompt, default, cast_func):
    user_input = input(prompt)
    if user_input.strip() == "":
        print(f"Using default value: {default}")
        return default
    try:
        return cast_func(user_input)
    except Exception as e:
        print(f"Invalid input. Using default value: {default}")
        return default

hidden_layer_sizes = get_user_input(
    "Enter hidden layer sizes separated by commas (e.g., 200,100,50): ", 
    [200, 100, 50], 
    lambda x: list(map(int, x.split(',')))
)
alpha = get_user_input("Enter the learning rate (e.g., 0.01): ", 0.01, float)
beta = get_user_input("Enter the momentum coefficient (e.g., 0.9): ", 0.9, float)
epochs = get_user_input("Enter the epochs: ", 100, int)

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
        image = x_test[:, idx].reshape(28, 28).T
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
