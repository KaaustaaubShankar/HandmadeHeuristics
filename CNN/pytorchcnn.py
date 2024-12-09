import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# FuzzyPooling Layer
class FuzzyPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(FuzzyPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x):
        # Get the shape of the input
        batch_size, channels, height, width = x.size()

        # Apply padding if needed
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Calculate the output shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # Initialize the output tensor
        output = torch.zeros(batch_size, channels, out_height, out_width, device=x.device)

        # Perform fuzzy pooling operation
        for i in range(out_height):
            for j in range(out_width):
                # Define the region for each kernel
                start_i = i * self.stride
                end_i = start_i + self.kernel_size
                start_j = j * self.stride
                end_j = start_j + self.kernel_size

                # Extract the region
                region = x[:, :, start_i:end_i, start_j:end_j]

                # Fuzzification (using a simple Gaussian membership function on the values in the region)
                fuzzified_region = self.fuzzify(region)

                # Ensure the fuzzified region has the correct shape before applying mean
                fuzzified_region = fuzzified_region.view(fuzzified_region.size(0), fuzzified_region.size(1), -1)  # Flatten the spatial part
                
                # Apply a "fuzzy average" over the flattened spatial part
                output[:, :, i, j] = fuzzified_region.mean(dim=-1)  # Mean over the flattened spatial part

        return output

    def fuzzify(self, region):
        # A simple fuzzification based on Gaussian function (higher values are more significant)
        region = region.reshape(region.size(0), region.size(1), -1)  # Flatten the spatial part
        membership = torch.exp(-region**2 / 2.0)  # Gaussian membership function
        return membership * region

# Simple CNN with FuzzyPooling Layer
class CNNWithFuzzyPooling(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNNWithFuzzyPooling, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = FuzzyPooling(kernel_size=2, stride=2)  # Use fuzzy pooling here
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4  # Karpathy's constant
batch_size = 64
num_epochs = 3

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNNWithFuzzyPooling(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to device (GPU/CPU)
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

# Check accuracy on training & test to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# Output accuracy on training and test set
print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}%")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}%")
