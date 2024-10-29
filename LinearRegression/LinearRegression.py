import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, loss_function='mse', features=1):
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        #pred = weights * inputs + bais
        self.weights = np.zeros(features)
        self.bias = 0.0  
        self.epochs = 0

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def train(self, inputs, targets, epochs, debug = False):
        loss_history = []
        for epoch in range(epochs):
            predictions = self.predict(inputs)
            loss = self.calculate_loss(predictions, targets)
            loss_history.append(loss)
            self.gradient_descent(inputs, targets, predictions)
            
            if (debug):
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        self.epochs += epochs
        return loss_history

    def gradient_descent(self, inputs, targets, predictions):
        n_samples = len(targets)
        
        if self.loss_function == 'mse':
            dw = (2 / n_samples) * np.dot(inputs.T, (predictions - targets))
            db = (2 / n_samples) * np.sum(predictions - targets)
        elif self.loss_function == 'mae':
            dw = (1 / n_samples) * np.dot(inputs.T, np.sign(predictions - targets))
            db = (1 / n_samples) * np.sum(np.sign(predictions - targets))
        else:
            raise ValueError("Unsupported loss function")

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def calculate_loss(self, predictions, targets):
        if self.loss_function == 'mse':
            return np.mean(np.square(predictions - targets))
        elif self.loss_function == 'mae':
            return np.mean(np.abs(predictions - targets))
        else:
            raise ValueError("Unsupported loss function")
    def printWeights(self):
        print(f"Trained {self.epochs} to achieve y={self.weights}x + {self.bias}")
    def reset(self):
        self.weights = np.zeros(len(self.weights))
        self.bias = 0.0  
        self.epochs = 0

