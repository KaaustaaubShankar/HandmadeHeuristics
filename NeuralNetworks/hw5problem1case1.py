import numpy as np
import matplotlib.pyplot as plt
from yuh import Layer
from Layers import train_test_split, mse_loss, mse_gradient
from yuh import forward_prop, train_with_momentum, compute_accuracy
from yuh import create_confusion_matrix, plot_confusion_matrix, plot_error_fractions

x_data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt')  
y_data = np.loadtxt('HW3_datafiles/MNISTnumLabels5000_balanced.txt')
x_train, x_test,y_train,y_test = train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True)


# shell of network
input_size = 784  
output_size = 10

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
    [200], 
    lambda x: list(map(int, x.split(',')))
)
alpha = get_user_input("Enter the learning rate (e.g., 0.01): ", 0.01, float)
beta = get_user_input("Enter the momentum coefficient (e.g., 0.9): ", 0.9, float)
epochs = get_user_input("Enter the epochs: ", 500, int)
batch_size = get_user_input("Enter the batch size: ", 128, int)

# build the network
layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
network = [
    Layer(layer_sizes[i], layer_sizes[i+1], activation='relu' if i < len(hidden_layer_sizes) else 'softmax')
    for i in range(len(layer_sizes) - 1)
]

# Load weights from autoencoder.npy
autoencoder_weights = np.load('autoencoder.npy', allow_pickle=True).item()
# Access the weights and biases for the first layer
network[0].W = autoencoder_weights['network_weights'][0]['W']
network[0].b = autoencoder_weights['network_weights'][0]['b']

# initial activations for initial error
initial_train_activations = forward_prop(x_train, network)
initial_test_activations = forward_prop(x_test, network)

initial_train_confusion_matrix = create_confusion_matrix(initial_train_activations[-1], y_train)
initial_test_confusion_matrix = create_confusion_matrix(initial_test_activations[-1], y_test)
plot_confusion_matrix(initial_train_confusion_matrix, initial_test_confusion_matrix)

initial_train_error = 1 - compute_accuracy(initial_train_activations[-1], y_train) / 100
initial_test_error = 1 - compute_accuracy(initial_test_activations[-1], y_test) / 100


initial_errors = {
    'train': initial_train_error,
    'test': initial_test_error
}

# train time
trainErrorFractions, testErrorFractions,epochs_trained = train_with_momentum(network, x_train, y_train, alpha, epochs, mse_loss, mse_gradient, beta, batch_size, verbose=True, X_test=x_test, Y_test=y_test, target_error=0.01)



# test time
train_activations = forward_prop(x_train, network)
train_error = 1 - compute_accuracy(train_activations[-1], y_train) / 100
trainErrorFractions.append(train_error)

test_activations = forward_prop(x_test, network)
test_error = 1 - compute_accuracy(test_activations[-1], y_test) / 100
testErrorFractions.append(test_error)

plot_error_fractions(trainErrorFractions, testErrorFractions)

# display the prediction
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

# Create confusion matrices
train_confusion_matrix = create_confusion_matrix(train_activations[-1], y_train)
test_confusion_matrix = create_confusion_matrix(test_activations[-1], y_test)

plot_confusion_matrix(train_confusion_matrix, test_confusion_matrix)






# Save final network weights and parameters
final_network_data = {
    'network_weights': [{'W': layer.W, 'b': layer.b} for layer in network],
    'hyperparameters': {
        'learning_rate': alpha,
        'momentum': beta
    }
}

# Calculate and save error for each test point
test_predictions = test_activations[-1]
individual_test_errors = []
for i in range(x_test.shape[1]):
    pred = test_predictions[:, i]
    true_label = y_test[i]
    pred_label = np.argmax(pred)
    error = 1 if pred_label != true_label else 0
    individual_test_errors.append(error)

final_network_data['test_errors'] = individual_test_errors

# Save all data to a numpy file

print("\nNetwork checkpoint saved successfully!")
print(f"Final test error rate: {test_error:.4f}")
print(f"Number of test samples: {len(individual_test_errors)}")
print(f"Number of misclassified samples: {sum(individual_test_errors)}")



# Calculate errors for each digit in training and test sets
def calculate_errors_by_digit(predictions, Y):
    digit_errors = {i: [] for i in range(10)}
    pred_labels = np.argmax(predictions, axis=0)
    
    for i in range(len(Y)):
        true_label = Y[i]
        error = 1 if pred_labels[i] != true_label else 0
        digit_errors[true_label].append(error)
    
    # Calculate mean error rate for each digit
    mean_errors = {digit: np.mean(errors) for digit, errors in digit_errors.items()}
    return mean_errors

# Calculate errors for training and test sets
train_errors_by_digit = calculate_errors_by_digit(train_activations[-1], y_train)
test_errors_by_digit = calculate_errors_by_digit(test_activations[-1], y_test)

# Plot overall error rates
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
overall_errors = [
    np.mean(list(train_errors_by_digit.values())), 
    np.mean(list(test_errors_by_digit.values()))
]
plt.bar(['Training', 'Test'], overall_errors, color=['blue', 'orange'])
plt.title('Overall Error Rates')
plt.ylabel('Error Rate')
plt.grid(True, alpha=0.3)

# Plot error rates by digit
plt.subplot(1, 2, 2)
digits = range(10)
width = 0.35
plt.bar([x - width/2 for x in digits], 
        list(train_errors_by_digit.values()), 
        width, 
        label='Training', 
        color='blue')
plt.bar([x + width/2 for x in digits], 
        list(test_errors_by_digit.values()), 
        width, 
        label='Test', 
        color='orange')
plt.xlabel('Digit')
plt.ylabel('Error Rate')
plt.title('Error Rates by Digit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(digits)

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nDetailed Error Statistics:")
print("\nTraining Error Rates by Digit:")
for digit, error in train_errors_by_digit.items():
    print(f"Digit {digit}: {error:.4f}")

print("\nTest Error Rates by Digit:")
for digit, error in test_errors_by_digit.items():
    print(f"Digit {digit}: {error:.4f}")

print(f"\nOverall Training Error Rate: {overall_errors[0]:.4f}")
print(f"Overall Test Error Rate: {overall_errors[1]:.4f}")
