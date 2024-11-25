# Imports
import numpy as np
import matplotlib.pyplot as plt
from Layers import Layer, mse_loss, mse_gradient
from yuh import train_test_split
from Layers import forward_prop, train_with_momentum
import math

# Load data
x_data = np.loadtxt('HW3_datafiles/MNISTnumImages5000_balanced.txt')
y_data = np.loadtxt('HW3_datafiles/MNISTnumLabels5000_balanced.txt')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True)

# Network architecture
input_size = 784
hidden_size = 200
output_size = 784

network = [
    Layer(input_size, hidden_size, activation='relu'),
    Layer(hidden_size, output_size, activation='sigmoid')
]

# Calculate metrics for each digit
def calculate_digit_metrics(network, X, Y, digit):
    digit_indices = np.where(Y == digit)[0]
    X_digit = X[:, digit_indices]

    print(X_digit.shape)
    
    reconstructions = forward_prop(X_digit, network)[-1]
    
    errors = np.sum((X_digit - reconstructions) ** 2, axis=0) / 2
    
    mre = np.mean(errors/100)
    std = np.std(errors/100)
    
    return mre, std

# Training parameters
epochs = 4000
alpha = 0.001
beta = 0.9
batch_size = 128

print(len(x_train))
# Train network
train_errors, test_errors, epochs_trained = train_with_momentum(
    network, x_train, x_train,
    alpha, epochs, mse_loss, mse_gradient, beta, 
    batch_size, verbose=True,
    X_test=x_test, Y_test=x_test,
    isClassification=False,
    target_error=0.01
)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, split_percentage=0.8, shuffle_train=True,unique_digits = np.array([0,1,2,3,4,5,6,7,8,9]))

# Print metrics for each digit
print("\nTest Set Metrics per Digit:")
print("Digit\tMRE\t\tStd Dev")
print("-" * 30)
for digit in range(10):
    mre, std = calculate_digit_metrics(network, x_test, y_test, digit)
    print(f"{digit}\t{mre:.6f}\t{std:.6f}")

# Plot original vs reconstructed images
def plot_reconstructions(network, X, Y, num_images=5):
    plt.figure(figsize=(15, 20))
    
    for digit_idx, digit in enumerate(range(5,10)):
        # Get indices for current digit
        digit_indices = np.where(Y == digit)[0]
        selected_indices = np.random.choice(digit_indices, num_images, replace=False)
        
        # Plot original images
        for i, idx in enumerate(selected_indices):
            plt.subplot(10, 5, digit_idx*10 + i + 1)
            plt.imshow(X[:, idx].reshape(28, 28).T, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(f'Original Digit {digit}')
        
        # Plot reconstructed images
        for i, idx in enumerate(selected_indices):
            plt.subplot(10, 5, digit_idx*10 + i + 6)
            reconstruction = forward_prop(X[:, idx].reshape(-1, 1), network)[-1]
            plt.imshow(reconstruction.reshape(28, 28).T, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(f'Reconstructed Digit {digit}')
    
    plt.tight_layout()
    plt.show()

plot_reconstructions(network, x_test,y_test)

# Plot final errors
plt.figure(figsize=(8, 6))
plt.bar(['Training MRE', 'Test MRE'], 
        [train_errors[-1], test_errors[-1]], 
        color=['blue', 'orange'])
plt.ylabel('Mean Reconstruction Error')
plt.title('Final Training vs Test MRE')
plt.grid(True, axis='y')
plt.show()

# Plot error over time
plt.figure(figsize=(12, 8), dpi=100)

epochs_range = np.arange(len(train_errors))
plt.plot(epochs_range, train_errors, 'b-', label='Training MRE', linewidth=2, alpha=0.8)
plt.plot(epochs_range, test_errors, 'r-', label='Test MRE', linewidth=2, alpha=0.8)

box_style = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8)

plt.annotate(f'Start: {train_errors[0]:.3f}', 
            xy=(epochs_range[0], train_errors[0]),
            xytext=(20, 20),
            textcoords='offset points',
            bbox=box_style,
            color='blue',
            fontsize=9)

plt.annotate(f'End: {train_errors[-1]:.3f}',
            xy=(epochs_range[-1], train_errors[-1]),
            xytext=(-20, 20),
            textcoords='offset points',
            bbox=box_style,
            color='blue',
            fontsize=9,
            ha='right')

plt.annotate(f'Start: {test_errors[0]:.3f}',
            xy=(epochs_range[0], test_errors[0]),
            xytext=(20, -30),
            textcoords='offset points',
            bbox=box_style,
            color='red',
            fontsize=9)

plt.annotate(f'End: {test_errors[-1]:.3f}',
            xy=(epochs_range[-1], test_errors[-1]),
            xytext=(-20, -30),
            textcoords='offset points',
            bbox=box_style,
            color='red',
            fontsize=9,
            ha='right')

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Reconstruction Error', fontsize=12)
plt.title('Training and Test MRE Over Time', fontsize=14, pad=20)

plt.legend([
    f'Training MRE (Final: {train_errors[-1]:.3f})',
    f'Test MRE (Final: {test_errors[-1]:.3f})'
], loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()



