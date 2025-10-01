# ML Interview Exercise: Imbalanced Classification

## Overview
This exercise tests a candidate's ability to handle **class imbalance** in binary classification using PyTorch. It covers fundamental ML concepts including weighted loss functions, training loops, and evaluation metrics.

## Files
- `test_exercise.py` - Template for the candidate (incomplete)
- `test_solution.py` - Reference solution
- `requirements.txt` - Dependencies

## Exercise Description

### Context
The candidate is given a binary classification problem with **imbalanced data** (20% positive class, 80% negative class). They need to:
1. Implement a weighted loss function to handle class imbalance
2. Complete the training loop
3. Implement model evaluation with appropriate metrics

### Time Allocation
- **Junior/Mid-level**: 30-45 minutes
- **Senior**: 20-30 minutes

---

## What to Evaluate

### 1. Technical Knowledge (40%)

**Weighted Loss Implementation:**
- ✅ Correctly computes pos_weight as `n_neg / n_pos`
- ✅ Uses `BCEWithLogitsLoss` (not BCELoss + Sigmoid)
- ✅ Properly converts to tensor

**Training Loop:**
- ✅ Sets model to training mode
- ✅ Zeros gradients before backward pass
- ✅ Correct order: forward → loss → backward → step

**Evaluation:**
- ✅ Sets model to eval mode
- ✅ Uses `torch.no_grad()` context
- ✅ Applies sigmoid to logits before thresholding
- ✅ Uses appropriate metrics (Precision, Recall, F1)

### 2. Problem-Solving Approach (30%)

**Good Signs:**
- Asks about class distribution before implementing loss
- Considers why weighted loss is needed for imbalanced data
- Tests code incrementally
- Checks tensor shapes/types

**Red Flags:**
- Jumps to coding without understanding the imbalance problem
- Uses accuracy as the primary metric
- Doesn't question why certain approaches are needed

### 3. Code Quality (20%)

- Clean, readable code
- Proper error handling (optional but nice)
- Comments where appropriate
- Follows existing code style

### 4. ML Fundamentals (10%)

**Discussion Points:**
- *Why use weighted loss instead of resampling?*
  - Answer: Both work, weighted loss is simpler; resampling can help with very extreme imbalance
  
- *Why F1/Precision/Recall instead of accuracy?*
  - Answer: Accuracy is misleading for imbalanced data (predicting all negative gives 80% accuracy!)
  
- *What does pos_weight do mathematically?*
  - Answer: Multiplies the loss for positive examples, making them contribute more to total loss

- *Alternative approaches?*
  - Answer: Oversampling (SMOTE), undersampling, focal loss, different thresholds, ensemble methods

---

## How to Conduct the Interview

### Setup (5 min before candidate joins)
```bash
# Send candidate the exercise file
# Ensure they have the environment set up
source venv/bin/activate
python test_exercise.py  # Should fail/produce errors
```

### Introduction (5 min)
1. Explain the context: "You have an imbalanced binary classification problem"
2. Show them the data distribution (20% positive)
3. Point them to the 3 TODOs
4. Let them know they can ask questions

### During the Exercise (20-40 min)
- **Observe their process**: Do they read the code first? Ask questions?
- **Minimal hints**: Only help if completely stuck
  - Stuck on TODO 1: "How do you tell PyTorch to weight certain examples more?"
  - Stuck on TODO 2: "What's the typical PyTorch training loop structure?"
  - Stuck on TODO 3: "What happens if you don't use sigmoid on the raw logits?"

### Evaluation (5-10 min)
Once they complete it:
```bash
python test_exercise.py
```

Expected output:
```
Epoch 0 | loss 0.XXXX
Epoch 1 | loss 0.XXXX
...
Epoch 4 | loss 0.XXXX
Precision 0.XXX Recall 0.XXX F1 0.XXX
```

### Discussion Questions (10 min)
1. "Your F1 is 0.78 with Recall 0.98. Is this good or bad?"
2. "What if we needed higher precision instead of recall?"
3. "How would you handle 1% positive class instead of 20%?"
4. "What are the limitations of your approach?"

---

## Scoring Rubric

| Criterion | Points | What to Look For |
|-----------|--------|------------------|
| **TODO 1: Weighted Loss** | 30 | Correct pos_weight calculation, proper loss function |
| **TODO 2: Training Loop** | 30 | Complete working loop, proper grad handling |
| **TODO 3: Evaluation** | 25 | Eval mode, no_grad, sigmoid, correct metrics |
| **Code Quality** | 10 | Readable, follows conventions |
| **Discussion** | 5 | Understanding of tradeoffs and alternatives |
| **Total** | 100 | |

**Passing Score**: 70+

---

## Common Mistakes to Watch For

### 🚨 Critical Errors
- Using `BCELoss` instead of `BCEWithLogitsLoss` (numerical instability)
- Not applying sigmoid before thresholding in evaluation
- Using accuracy as the metric
- Forgetting `zero_grad()`

### ⚠️ Red Flags
- Not understanding why weighted loss is needed
- Can't explain precision vs recall tradeoff
- Hardcoding values without understanding them

### ✅ Bonus Points
- Suggests stratified train/test split
- Mentions threshold tuning as alternative
- Discusses precision-recall curve or ROC-AUC
- Asks about business context (is false positive or false negative worse?)

---

## Variations for Different Levels

### Junior Level
- Provide more hints in comments
- Pre-fill some parts (e.g., model.train(), model.eval())
- Focus on getting it working

### Senior Level
- Remove more scaffolding
- Add requirement: "Also implement class-weighted random sampling as alternative"
- Discussion: "Design a complete pipeline for production"

### ML Engineer/Researcher
- "The model isn't performing well. Debug and improve it"
- "Implement focal loss as an alternative"
- "Add experiment tracking (wandb/mlflow)"

---

## Follow-up Challenges

If they finish early or you want to extend:

1. **Threshold Tuning**: "Find the optimal threshold for F1 score"
2. **Oversampling**: "Try SMOTE instead of weighted loss"
3. **Model Improvement**: "Double the F1 score - how would you do it?"
4. **Production**: "How would you deploy this? What could go wrong?"

---

## Key Takeaways

This exercise reveals:
- ✅ PyTorch fundamentals (training loop, loss functions)
- ✅ Understanding of class imbalance
- ✅ Metric selection for imbalanced problems
- ✅ Practical ML engineering skills
- ✅ Problem-solving under constraints
