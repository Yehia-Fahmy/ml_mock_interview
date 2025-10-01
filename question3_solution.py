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
        """Initialize a simple 2-layer neural network"""
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
        """Sigmoid activation function with numerical stability"""
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a1, self.a2
    
    def backward(self, X, y, a1, a2):
        """Backward propagation"""
        m = X.shape[0]  # Number of samples
        
        # Output layer error (for binary classification)
        dz2 = a2 - y.reshape(-1, 1)
        
        # Hidden layer error
        dz1 = dz2 @ self.W2.T * self.sigmoid_derivative(self.z1)
        
        # Compute gradients
        dW2 = (1/m) * a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, gradients):
        """Update parameters using gradient descent"""
        dW1, db1, dW2, db2 = gradients
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            a1, a2 = self.forward(X)
            
            # Compute loss (mean squared error)
            loss = np.mean((a2.flatten() - y) ** 2)
            losses.append(loss)
            
            # Backward propagation
            gradients = self.backward(X, y, a1, a2)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Print progress every 100 epochs
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        _, a2 = self.forward(X)
        return a2.flatten()
    
    def predict_classes(self, X, threshold=0.5):
        """Predict binary classes"""
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)

# ---- Train and Evaluate the Network ----
print("=== Neural Network Training ===")
nn = SimpleNeuralNetwork(input_size=X_train_scaled.shape[1], hidden_size=8, output_size=1)
losses = nn.train(X_train_scaled, y_train, epochs=1000)

print("\n=== Model Evaluation ===")
y_pred = nn.predict_classes(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n=== Learning Curve ===")
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("\n=== Key Concepts Demonstrated ===")
print("- Forward propagation: Computing predictions layer by layer")
print("- Backward propagation: Computing gradients using chain rule")
print("- Gradient descent: Updating parameters to minimize loss")
print("- Activation functions: Sigmoid for non-linear decision boundaries")
print("- Weight initialization: Small random values to break symmetry")

# =============================================================================
# INSTRUCTOR HINTS FOR QUESTION 3: SIMPLE NEURAL NETWORK FROM SCRATCH
# =============================================================================
# If the candidate gets stuck, here are progressive hints you can give:
#
# TODO 1 - Forward Propagation:
# Hint 1: "Start with linear transformation: z = X @ W + b"
# Hint 2: "Apply activation function: a = sigmoid(z)"
# Hint 3: "For sigmoid: 1 / (1 + exp(-x)), handle numerical stability"
# Hint 4: "Store intermediate values (z1, a1, z2, a2) for backprop"
#
# TODO 2 - Backward Propagation:
# Hint 1: "Start with output layer error: dz2 = a2 - y"
# Hint 2: "Use chain rule: dz1 = dz2 @ W2.T * sigmoid_derivative(z1)"
# Hint 3: "Compute gradients: dW = (1/m) * X.T @ dz"
# Hint 4: "Don't forget to average over batch size (1/m)"
#
# TODO 3 - Training:
# Hint 1: "Initialize weights with small random values"
# Hint 2: "Use mean squared error: loss = mean((pred - true)²)"
# Hint 3: "Update parameters: W = W - learning_rate * dW"
# Hint 4: "Track loss over epochs to see convergence"
#
# General Hints:
# - "Why do we need activation functions in neural networks?"
# - "What happens if you initialize all weights to zero?"
# - "How does the learning rate affect training speed and stability?"
# - "What's the difference between forward and backward propagation?"