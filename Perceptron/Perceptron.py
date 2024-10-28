import numpy as np

class Perceptron:
    def __init__(self, input_size=784):
        self.input_size = input_size
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = np.random.uniform(-0.5, 0.5)  # Separate bias term

    def forward(self, image):
        net_input = np.dot(self.weights, image) + self.bias
        
        # Apply the activation function (step function)
        output = 1 if net_input > 0 else 0
        
        return output
