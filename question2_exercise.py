import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---- Synthetic Data Generation ----
def generate_data(n_samples=200, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data with some irrelevant features"""
    np.random.seed(random_state)
    
    # Create true coefficients (only first 5 features are relevant)
    true_coef = np.zeros(n_features)
    true_coef[:5] = np.random.normal(0, 2, 5)  # First 5 features matter
    
    # Generate features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate target with noise
    y = X @ true_coef + np.random.normal(0, noise, n_samples)
    
    return X, y, true_coef

X, y, true_coef = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
print(f"Test set: {X_test_scaled.shape[0]} samples")
print(f"True coefficients (first 5): {true_coef[:5]}")

# ---- TODO 1: Implement Ridge Regression (L2 Regularization) ----
# Ridge regression adds L2 penalty: ||w||² to the loss function
# 
# The closed-form solution is: w = (X^T X + αI)^(-1) X^T y
# where α (alpha) is the regularization strength
#
# Implement the Ridge regression class below:

class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        Initialize Ridge regression
        
        Args:
            alpha (float): Regularization strength. Higher alpha = more regularization
        """
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit Ridge regression model
        
        Args:
            X (np.array): Training features (n_samples, n_features)
            y (np.array): Training targets (n_samples,)
        """
        # Add intercept term (bias)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # YOUR CODE HERE:
        # Implement Ridge regression closed-form solution
        # Hint: w = (X^T X + αI)^(-1) X^T y
        # Remember to handle the intercept term properly
        
        pass  # Replace with your implementation
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Features (n_samples, n_features)
        
        Returns:
            np.array: Predictions (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # YOUR CODE HERE:
        # Implement prediction: y = X @ coef + intercept
        pass  # Replace with your implementation

# Test Ridge regression
ridge = RidgeRegression(alpha=1.0)
# ridge.fit(X_train_scaled, y_train)
# ridge_pred = ridge.predict(X_test_scaled)

# ---- TODO 2: Implement Lasso Regression (L1 Regularization) ----
# Lasso regression adds L1 penalty: ||w||₁ to the loss function
# Unlike Ridge, Lasso has no closed-form solution and requires optimization
# 
# Implement coordinate descent algorithm for Lasso:

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Initialize Lasso regression
        
        Args:
            alpha (float): Regularization strength
            max_iter (int): Maximum iterations for coordinate descent
            tol (float): Convergence tolerance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def _soft_threshold(self, x, threshold):
        """
        Soft thresholding function for Lasso
        S(x, λ) = sign(x) * max(|x| - λ, 0)
        """
        # YOUR CODE HERE:
        # Implement soft thresholding function
        pass  # Replace with your implementation
    
    def fit(self, X, y):
        """
        Fit Lasso regression using coordinate descent
        
        Args:
            X (np.array): Training features (n_samples, n_features)
            y (np.array): Training targets (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)
        
        # Center the data
        y_centered = y - self.intercept_
        
        # YOUR CODE HERE:
        # Implement coordinate descent algorithm
        # For each iteration:
        #   1. For each feature j:
        #      - Compute residual: r = y_centered - X @ coef + coef[j] * X[:, j]
        #      - Update coef[j] using soft thresholding
        #   2. Check for convergence
        
        pass  # Replace with your implementation
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Features (n_samples, n_features)
        
        Returns:
            np.array: Predictions (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # YOUR CODE HERE:
        # Implement prediction: y = X @ coef + intercept
        pass  # Replace with your implementation

# Test Lasso regression
lasso = LassoRegression(alpha=0.1)
# lasso.fit(X_train_scaled, y_train)
# lasso_pred = lasso.predict(X_test_scaled)

# ---- TODO 3: Compare Performance and Interpret Results ----
# Compare Ridge vs Lasso on the test set and analyze the results

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    return mse, r2

# YOUR CODE HERE:
# 1. Train both Ridge and Lasso models
# 2. Make predictions on test set
# 3. Evaluate both models
# 4. Compare the learned coefficients
# 5. Interpret the results (which features were selected by Lasso?)

print("\n=== Model Comparison ===")
# Uncomment and complete the evaluation:
# ridge_mse, ridge_r2 = evaluate_model(y_test, ridge_pred, "Ridge")
# lasso_mse, lasso_r2 = evaluate_model(y_test, lasso_pred, "Lasso")

print("\n=== Coefficient Analysis ===")
# print("True coefficients (first 5):", true_coef[:5])
# print("Ridge coefficients (first 5):", ridge.coef_[:5])
# print("Lasso coefficients (first 5):", lasso.coef_[:5])

# ---- Expected Output ----
# After successful implementation, you should see:
# - Both models achieving reasonable R² scores (>0.7)
# - Lasso coefficients showing sparsity (some exactly zero)
# - Ridge coefficients being shrunk but not zero
# - Discussion of the trade-offs between L1 and L2 regularization