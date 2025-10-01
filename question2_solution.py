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

# ---- Ridge Regression (L2 Regularization) ----
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit Ridge regression model using closed-form solution"""
        # Add intercept term (bias)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Ridge regression closed-form solution: w = (X^T X + αI)^(-1) X^T y
        # Create regularization matrix (don't regularize intercept)
        reg_matrix = np.eye(X_with_intercept.shape[1])
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Compute coefficients
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y
        self.coef_full = np.linalg.solve(XTX + self.alpha * reg_matrix, XTy)
        
        # Separate intercept and coefficients
        self.intercept_ = self.coef_full[0]
        self.coef_ = self.coef_full[1:]
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.coef_ + self.intercept_

# ---- Lasso Regression (L1 Regularization) ----
class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding function for Lasso"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        """Fit Lasso regression using coordinate descent"""
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)
        
        # Center the data
        y_centered = y - self.intercept_
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                # Compute residual without feature j
                residual = y_centered - X @ self.coef_ + self.coef_[j] * X[:, j]
                
                # Update coefficient j using soft thresholding
                numerator = X[:, j] @ residual
                denominator = X[:, j] @ X[:, j]
                
                if denominator > 0:
                    self.coef_[j] = self._soft_threshold(numerator / denominator, 
                                                       self.alpha / denominator)
            
            # Check convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.coef_ + self.intercept_

# ---- Model Training and Evaluation ----
def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    return mse, r2

# Train models
ridge = RidgeRegression(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)

lasso = LassoRegression(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)

# Evaluate models
print("\n=== Model Comparison ===")
ridge_mse, ridge_r2 = evaluate_model(y_test, ridge_pred, "Ridge")
lasso_mse, lasso_r2 = evaluate_model(y_test, lasso_pred, "Lasso")

print("\n=== Coefficient Analysis ===")
print("True coefficients (first 5):", true_coef[:5])
print("Ridge coefficients (first 5):", ridge.coef_[:5])
print("Lasso coefficients (first 5):", lasso.coef_[:5])

print("\n=== Feature Selection Analysis ===")
print(f"Features selected by Lasso (non-zero): {np.sum(np.abs(lasso.coef_) > 1e-6)}")
print(f"Features selected by Ridge (non-zero): {np.sum(np.abs(ridge.coef_) > 1e-6)}")

print("\n=== Interpretation ===")
print("- Ridge shrinks coefficients but keeps all features")
print("- Lasso performs feature selection (sets some coefficients to exactly zero)")
print("- Both help prevent overfitting, but Lasso is better for feature selection")

# =============================================================================
# INSTRUCTOR HINTS FOR QUESTION 2: LINEAR REGRESSION WITH REGULARIZATION
# =============================================================================
# If the candidate gets stuck, here are progressive hints you can give:
#
# TODO 1 - Ridge Regression:
# Hint 1: "Ridge has a closed-form solution - no optimization needed"
# Hint 2: "The formula is: w = (X^T X + αI)^(-1) X^T y"
# Hint 3: "Don't forget to add the intercept term to X"
# Hint 4: "Use np.linalg.solve() for the matrix inversion"
#
# TODO 2 - Lasso Regression:
# Hint 1: "Lasso requires optimization - use coordinate descent"
# Hint 2: "For each feature j, update: w_j = soft_threshold(X_j^T r, α)"
# Hint 3: "Soft thresholding: S(x,λ) = sign(x) * max(|x| - λ, 0)"
# Hint 4: "Update features one at a time in a loop"
#
# TODO 3 - Comparison:
# Hint 1: "Compare MSE and R² scores on test data"
# Hint 2: "Look at which coefficients are exactly zero in Lasso"
# Hint 3: "Ridge shrinks all coefficients, Lasso sets some to zero"
# Hint 4: "Which approach would you use for feature selection?"
#
# General Hints:
# - "What's the difference between L1 and L2 penalties?"
# - "Why does Lasso perform feature selection while Ridge doesn't?"
# - "How do you choose the regularization parameter α?"