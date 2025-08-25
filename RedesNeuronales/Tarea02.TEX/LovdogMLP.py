import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

class BasicMLP:
    def __init__(self, layer_sizes=(10, 5), max_iter=1000):
        """
        Simple MLP class with sigmoid activation
        
        Args:
            layer_sizes: tuple of hidden layer sizes, e.g., (10, 5) for two layers
            max_iter: maximum training iterations
        """
        self.layer_sizes = layer_sizes
        self.max_iter = max_iter
        self.model = None
        self.training_loss = []
        
    def create_model(self):
        """Create the MLP model with specified architecture"""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.layer_sizes,
            activation='logistic',  # Sigmoid activation
            max_iter=self.max_iter,
            random_state=42,
            verbose=False
        )
        return self.model
    
    def train(self, X, y):
        """
        Train the MLP on X, y data
        
        Args:
            X: input features (2D array)
            y: target labels (1D array)
        """
        if self.model is None:
            self.create_model()
        
        # Train and capture loss curve
        self.model.fit(X, y)
        self.training_loss = self.model.loss_curve_
        
        return self.training_loss
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def get_accuracy(self, X, y):
        """Calculate accuracy on given data"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def save_training_stats(self, filename='training_stats.txt'):
        """Save training statistics to a file"""
        if not self.training_loss:
            print("No training statistics available")
            return
        
        with open(filename, 'w') as f:
            f.write(f"MLP Training Statistics\n")
            f.write(f"=======================\n")
            f.write(f"Layer architecture: {self.layer_sizes}\n")
            f.write(f"Final loss: {self.training_loss[-1]:.6f}\n")
            f.write(f"Training iterations: {len(self.training_loss)}\n")
            f.write(f"\nLoss per iteration:\n")
            for i, loss in enumerate(self.training_loss):
                f.write(f"Iteration {i+1}: {loss:.6f}\n")
        
        print(f"Training statistics saved to {filename}")
    
    def plot_training_loss(self):
        """Plot the training loss curve"""
        if not self.training_loss:
            print("No training data available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_loss, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'MLP Training Loss (Layers: {self.layer_sizes})')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            return "Model not trained yet"
        
        info = f"""
        MLP Model Information:
        ---------------------
        Layers: Input -> {self.layer_sizes} -> Output
        Activation: Sigmoid
        Final Loss: {self.training_loss[-1] if self.training_loss else 'N/A':.6f}
        Number of iterations: {len(self.training_loss) if self.training_loss else 0}
        """
        return info

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your own)
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [0.5, 0.5], [0.2, 0.8]])
    y = np.array([0, 1, 1, 0, 0, 1])
    
    # Create MLP with different architectures
    print("Testing different MLP architectures:")
    
    # Single hidden layer
    mlp1 = BasicMLP(layer_sizes=(8,), max_iter=500)
    mlp1.train(X, y)
    print(mlp1.get_model_info())
    mlp1.plot_training_loss()
    mlp1.save_training_stats('mlp1_stats.txt')
    
    # Two hidden layers
    mlp2 = BasicMLP(layer_sizes=(10, 5), max_iter=500)
    mlp2.train(X, y)
    print(mlp2.get_model_info())
    
    # Three hidden layers
    mlp3 = BasicMLP(layer_sizes=(15, 10, 5), max_iter=500)
    mlp3.train(X, y)
    print(mlp3.get_model_info())
    
    # Make predictions
    test_data = np.array([[0.3, 0.7], [0.8, 0.2]])
    predictions = mlp1.predict(test_data)
    print(f"Predictions: {predictions}")
