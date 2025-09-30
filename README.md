# Imbalanced Classification Interview Exercise

## Quick Start

### For Interviewers
1. **Review** `INTERVIEW_GUIDE.md` for full instructions
2. **Send** `test_exercise.py` to candidate
3. **Keep** `test_solution.py` as reference
4. **Expected time**: 20-45 minutes depending on level

### For Candidates
Complete the 3 TODOs in `test_exercise.py`:
1. Implement weighted BCE loss for handling class imbalance
2. Implement the training loop
3. Implement model evaluation

## File Structure
```
.
├── README.md                  # This file
├── INTERVIEW_GUIDE.md         # Complete interview guide for interviewers
├── requirements.txt           # Python dependencies
├── test.py                    # Original complete implementation
├── test_exercise.py           # Exercise template (incomplete)
└── test_solution.py           # Reference solution
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run exercise (will fail until completed)
python test_exercise.py

# Run solution to see expected output
python test_solution.py
```

## What's Being Tested

### Technical Skills
- ✅ PyTorch fundamentals (loss functions, training loop)
- ✅ Handling imbalanced datasets
- ✅ Proper evaluation metrics
- ✅ Understanding of classification thresholds

### Conceptual Understanding
- Why weighted loss for imbalanced data?
- When to use F1/Precision/Recall vs Accuracy?
- Tradeoffs between different approaches

## Expected Output (Solution)

```
Epoch 0 | loss 0.7761
Epoch 1 | loss 0.8496
Epoch 2 | loss 0.8517
Epoch 3 | loss 0.5683
Epoch 4 | loss 0.4219
Precision 0.651 Recall 0.983 F1 0.783
```

The key observation: High recall (0.983) shows the weighted loss successfully helps the model catch most positive examples despite class imbalance.

## Key Concepts

### Class Imbalance
The synthetic dataset is imbalanced (20% positive, 80% negative). Without handling this:
- Model might predict all negatives → 80% accuracy but useless
- Need weighted loss or resampling to balance learning

### Weighted BCE Loss
```python
pos_weight = n_negative / n_positive  # ≈ 4.0 in this case
```
This tells the model: "A mistake on a positive example is 4× worse than on a negative example"

### Metrics for Imbalanced Data
- **Accuracy**: Misleading (80% by predicting all negative!)
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we catch?
- **F1**: Harmonic mean of precision and recall

## Notes

- Original `test.py` had the TODO comment but was already implemented
- This has been restructured as a proper interview exercise
- See `INTERVIEW_GUIDE.md` for complete evaluation rubric and discussion questions