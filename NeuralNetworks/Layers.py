import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class SoftMax:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        exp_x = np.exp(x - np.max(x, axis=0))
        return exp_x / np.sum(exp_x, axis=0)

    def backward(self, grad_output):
        exp_x = np.exp(self.input - np.max(self.input, axis=0))
        softmax = exp_x / np.sum(exp_x, axis=0)
        return grad_output * softmax * (1 - softmax)

class MSE:
    def __init__(self):
        self.input = None
        self.target = None

    def forward(self, x, target):
        self.input = x
        self.target = target
        return np.mean((x - target) ** 2)

    def backward(self):
        return 2 * (self.input - self.target) / self.input.size