import numpy as np
import matplotlib.pyplot as plt
from Layers import Layer, train_test_split, mse_loss, mse_gradient
from Layers import forward_prop, train_with_momentum

# Load and prepare data
x_data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt')
y_data = np.loadtxt('HW3_datafiles/MNISTnumLabels5000_balanced.txt')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True)

# Network configuration
input_size = 784
hidden_size = 200  # Same as Problem 1's hidden layer
output_size = 784  # Same as input for reconstruction

# Create autoencoder network
network = [
    Layer(input_size, hidden_size, activation='relu'),  # Encoder
    Layer(hidden_size, output_size, activation='sigmoid')  # Decoder
]

def calculate_digit_metrics(network, X, Y, digit):
    """Calculate MRE and standard deviation for a specific digit"""
    digit_indices = np.where(Y == digit)[0]
    X_digit = X[:, digit_indices]
    
    # Get reconstructions
    reconstructions = forward_prop(X_digit, network)[-1]
    
    # Calculate reconstruction errors for each sample
    errors = np.sum((X_digit - reconstructions) ** 2, axis=0) / 2
    
    # Calculate mean and standard deviation
    mre = np.mean(errors)
    std = np.std(errors)
    
    return mre, std

# Train the autoencoder
epochs = 100
alpha = 0.2
beta = 0.9
batch_size = 128

# Train using reconstruction error (MSE)
train_errors, test_errors = train_with_momentum(
    network, x_train, x_train,  # Remove the .T here
    alpha, epochs, mse_loss, mse_gradient, beta,
    batch_size, verbose=True,
    X_test=x_test, Y_test=x_test,  # Remove the .T here
    isClassification=False
)

# Calculate metrics for each digit
print("\nTest Set Metrics per Digit:")
print("Digit\tMRE\t\tStd Dev")
print("-" * 30)
for digit in range(10):
    mre, std = calculate_digit_metrics(network, x_test, y_test, digit)
    print(f"{digit}\t{mre:.6f}\t{std:.6f}")

# Visualize original vs reconstructed images
def plot_reconstructions(network, X, num_images=5):
    indices = np.random.choice(X.shape[1], num_images, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, idx].reshape(28, 28).T, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        
        # Reconstructed image
        plt.subplot(2, num_images, i + 1 + num_images)
        reconstruction = forward_prop(X[:, idx].reshape(-1, 1), network)[-1]
        plt.imshow(reconstruction.reshape(28, 28).T, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.show()

# Plot some example reconstructions
plot_reconstructions(network, x_test)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(range(0, epochs, 10), train_errors, label='Training MRE')
plt.plot(range(0, epochs, 10), test_errors, label='Test MRE')
plt.xlabel('Epoch')
plt.ylabel('Mean Reconstruction Error')
plt.title('Autoencoder Training Progress')
plt.legend()
plt.grid(True)
plt.show()
