# ML Interview Guide: 3 Entry-Level Questions

## Overview
Three progressive interview questions testing fundamental ML concepts suitable for entry-level ML engineers.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Question 1: Imbalanced Classification
**Time**: 30-45 minutes | **Difficulty**: Entry-level

### Files
- `question1_exercise.py` - Exercise template (incomplete)
- `question1_solution.py` - Reference solution

### Exercise
Complete 3 TODOs:
1. Implement weighted BCE loss for class imbalance
2. Complete training loop  
3. Implement evaluation

### Scoring (100 points)
- **TODO 1**: 30 points - Weighted loss implementation
- **TODO 2**: 30 points - Training loop
- **TODO 3**: 25 points - Evaluation
- **Code Quality**: 10 points
- **Discussion**: 5 points

### Key Evaluation Points
- ✅ Correct `pos_weight = n_neg / n_pos`
- ✅ Uses `BCEWithLogitsLoss` (not `BCELoss`)
- ✅ Proper training loop: `train()`, `zero_grad()`, `backward()`, `step()`
- ✅ Evaluation: `eval()`, `no_grad()`, sigmoid before thresholding
- ✅ Uses Precision/Recall/F1 (not accuracy)

### Expected Output
```
Epoch 0 | loss 0.9252
Epoch 1 | loss 0.7132
Epoch 2 | loss 0.6163
Epoch 3 | loss 0.5111
Epoch 4 | loss 0.4285
Precision 0.607 Recall 0.962 F1 0.744
```

### Discussion Questions
1. "Why F1/Precision/Recall instead of accuracy?"
2. "What if we needed higher precision instead of recall?"
3. "How would you handle 1% positive class instead of 20%?"

---

## Question 2: Linear Regression with Regularization
**Time**: 25-35 minutes | **Difficulty**: Entry-level

### Files
- `question2_exercise.py` - Exercise template (incomplete)
- `question2_solution.py` - Reference solution

### Exercise
Complete 3 TODOs:
1. Implement Ridge regression (L2 regularization)
2. Implement Lasso regression (L1 regularization)
3. Compare performance and interpret results

### Scoring (100 points)
- **TODO 1**: 35 points - Ridge regression implementation
- **TODO 2**: 35 points - Lasso regression implementation
- **TODO 3**: 20 points - Performance comparison
- **Code Quality**: 7 points
- **Discussion**: 3 points

### Key Evaluation Points
- ✅ Ridge: Closed-form solution `w = (X^T X + αI)^(-1) X^T y`
- ✅ Lasso: Coordinate descent with soft thresholding
- ✅ Proper handling of intercept term
- ✅ Understanding of L1 vs L2 regularization differences
- ✅ Feature selection interpretation

### Expected Output
```
Training set: 140 samples, 10 features
Test set: 60 samples
True coefficients (first 5): [ 0.99342831 -0.2765286   1.29537708  3.04605971 -0.46830675]

=== Model Comparison ===
Ridge:
  MSE: 0.0137
  R²: 0.9989
Lasso:
  MSE: 0.0126
  R²: 0.9990

=== Coefficient Analysis ===
True coefficients (first 5): [ 0.99342831 -0.2765286   1.29537708  3.04605971 -0.46830675]
Ridge coefficients (first 5): [ 0.99248102 -0.28616848  1.06447907  2.65292487 -0.45384691]
Lasso coefficients (first 5): [ 0.99844162 -0.28947549  1.07087111  2.6711988  -0.45670978]
```

### Discussion Questions
1. "When would you use Ridge vs Lasso?"
2. "Why does Lasso perform feature selection while Ridge doesn't?"
3. "How do you choose the regularization parameter α?"

---

## Question 3: Simple Neural Network from Scratch
**Time**: 35-45 minutes | **Difficulty**: Entry-level

### Files
- `question3_exercise.py` - Exercise template (incomplete)
- `question3_solution.py` - Reference solution

### Exercise
Complete 3 TODOs:
1. Implement forward propagation
2. Implement backpropagation
3. Train the network and evaluate

### Scoring (100 points)
- **TODO 1**: 30 points - Forward propagation
- **TODO 2**: 40 points - Backpropagation
- **TODO 3**: 20 points - Training and evaluation
- **Code Quality**: 7 points
- **Discussion**: 3 points

### Key Evaluation Points
- ✅ Correct sigmoid activation and derivative
- ✅ Forward pass: linear transformation + activation
- ✅ Backward pass: chain rule application
- ✅ Gradient descent parameter updates
- ✅ Proper weight initialization

### Expected Output
```
Training set: 700 samples, 4 features
Test set: 300 samples
Class distribution: [357 343]
=== Neural Network Training ===
Epoch 0, Loss: 0.2529
Epoch 100, Loss: 0.2506
Epoch 200, Loss: 0.2498
...
Epoch 900, Loss: 0.2467

=== Model Evaluation ===
Test Accuracy: 0.6433

Classification Report:
              precision    recall  f1-score   support
           0       0.56      0.99      0.72       138
           1       0.97      0.35      0.52       162
    accuracy                           0.64       300
```

### Discussion Questions
1. "Why do we need activation functions?"
2. "What happens if we initialize all weights to zero?"
3. "How does the learning rate affect training?"
