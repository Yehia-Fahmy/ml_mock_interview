import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ---- Data Generation ----
def generate_data(n_samples=1000, n_features=4, n_classes=2, random_state=42):
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        random_state=random_state
    )
    return X, y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
print(f"Test set: {X_test_scaled.shape[0]} samples")
print(f"Class distribution: {np.bincount(y_train)}")

# ---- Neural Network Implementation ----
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize a simple 2-layer neural network
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            output_size (int): Number of output classes
            learning_rate (float): Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases with small random values
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # YOUR CODE HERE:
        # Implement sigmoid: 1 / (1 + exp(-x))
        # Handle numerical stability for large negative values
        pass  # Replace with your implementation
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        # YOUR CODE HERE:
        # Implement sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        pass  # Replace with your implementation
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X (np.array): Input features (n_samples, n_features)
        
        Returns:
            tuple: (hidden_layer_output, output_layer_output)
        """
        # YOUR CODE HERE:
        # Implement forward propagation:
        # 1. Compute hidden layer: z1 = X @ W1 + b1, a1 = sigmoid(z1)
        # 2. Compute output layer: z2 = a1 @ W2 + b2, a2 = sigmoid(z2)
        # Return both a1 and a2 for backpropagation
        
        pass  # Replace with your implementation
    
    def backward(self, X, y, a1, a2):
        """
        Backward propagation
        
        Args:
            X (np.array): Input features (n_samples, n_features)
            y (np.array): True labels (n_samples,)
            a1 (np.array): Hidden layer activations
            a2 (np.array): Output layer activations
        
        Returns:
            tuple: Gradients for all parameters
        """
        m = X.shape[0]  # Number of samples
        
        # YOUR CODE HERE:
        # Implement backpropagation:
        # 1. Compute output layer error: dz2 = a2 - y (for binary classification)
        # 2. Compute hidden layer error: dz1 = dz2 @ W2.T * sigmoid_derivative(z1)
        # 3. Compute gradients:
        #    - dW2 = (1/m) * a1.T @ dz2
        #    - db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        #    - dW1 = (1/m) * X.T @ dz1
        #    - db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        pass  # Replace with your implementation
    
    def update_parameters(self, gradients):
        """
        Update parameters using gradient descent
        
        Args:
            gradients (tuple): Gradients for all parameters (dW1, db1, dW2, db2)
        """
        dW1, db1, dW2, db2 = gradients
        
        # YOUR CODE HERE:
        # Update all parameters using gradient descent:
        # parameter = parameter - learning_rate * gradient
        
        pass  # Replace with your implementation
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
        """
        losses = []
        
        for epoch in range(epochs):
            # YOUR CODE HERE:
            # 1. Forward propagation
            # 2. Compute loss (mean squared error for simplicity)
            # 3. Backward propagation
            # 4. Update parameters
            # 5. Store loss for plotting
            
            pass  # Replace with your implementation
            
            # Print progress every 100 epochs
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Predicted probabilities
        """
        # YOUR CODE HERE:
        # Use forward propagation to get predictions
        # Return probabilities (sigmoid output)
        
        pass  # Replace with your implementation
    
    def predict_classes(self, X, threshold=0.5):
        """
        Predict binary classes
        
        Args:
            X (np.array): Input features
            threshold (float): Classification threshold
        
        Returns:
            np.array: Predicted classes (0 or 1)
        """
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)

# ---- TODO: Train and Evaluate the Network ----
# Create and train a neural network

# YOUR CODE HERE:
# 1. Create a SimpleNeuralNetwork instance
# 2. Train it on the training data
# 3. Make predictions on test data
# 4. Evaluate accuracy and print classification report

print("=== Neural Network Training ===")
# Uncomment and complete:
# nn = SimpleNeuralNetwork(input_size=X_train_scaled.shape[1], hidden_size=8, output_size=1)
# losses = nn.train(X_train_scaled, y_train, epochs=1000)

print("\n=== Model Evaluation ===")
# y_pred = nn.predict_classes(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

print("\n=== Learning Curve ===")
# plt.figure(figsize=(10, 6))
# plt.plot(losses)
# plt.title('Training Loss Over Time')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# ---- Expected Output ----
# After successful implementation, you should see:
# - Training loss decreasing over epochs
# - Test accuracy around 0.85-0.95
# - Smooth learning curve showing convergence
# - Understanding of forward/backward propagation