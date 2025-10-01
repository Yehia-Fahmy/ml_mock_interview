# ML Interview Guide

## Overview
Binary classification with imbalanced data (20% positive, 80% negative). Tests PyTorch skills and ML fundamentals.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Exercise
Complete 3 TODOs in `test_exercise.py`:
1. Implement weighted BCE loss
2. Complete training loop  
3. Implement evaluation

## Files
- `test_exercise.py` - Exercise template (incomplete)
- `test_solution.py` - Reference solution

## Time Allocation
- **Junior**: 30-45 minutes
- **Senior**: 20-30 minutes

## Scoring (100 points)
- **TODO 1**: 30 points - Weighted loss implementation
- **TODO 2**: 30 points - Training loop
- **TODO 3**: 25 points - Evaluation
- **Code Quality**: 10 points
- **Discussion**: 5 points

**Passing**: 70+

## Key Evaluation Points

### Technical Knowledge
- ✅ Correct `pos_weight = n_neg / n_pos`
- ✅ Uses `BCEWithLogitsLoss` (not `BCELoss`)
- ✅ Proper training loop: `train()`, `zero_grad()`, `backward()`, `step()`
- ✅ Evaluation: `eval()`, `no_grad()`, sigmoid before thresholding
- ✅ Uses Precision/Recall/F1 (not accuracy)

### Red Flags
- ❌ Uses accuracy as primary metric
- ❌ Doesn't understand why weighted loss is needed
- ❌ Uses `BCELoss` instead of `BCEWithLogitsLoss`

## Discussion Questions
1. "Why F1/Precision/Recall instead of accuracy?"
2. "What if we needed higher precision instead of recall?"
3. "How would you handle 1% positive class instead of 20%?"

## Expected Output
```
Epoch 0 | loss 0.7761
...
Precision 0.651 Recall 0.983 F1 0.783
```
