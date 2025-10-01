# ML Interview: Imbalanced Classification

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Exercise

Complete 3 TODOs in `test_exercise.py`:
1. Implement weighted BCE loss for class imbalance
2. Implement training loop
3. Implement evaluation

## Expected Output

```
Epoch 0 | loss 0.7761
Epoch 1 | loss 0.8496
...
Precision 0.651 Recall 0.983 F1 0.783
```

## Key Concepts

- **Class Imbalance**: 20% positive, 80% negative classes
- **Weighted Loss**: `pos_weight = n_negative / n_positive`
- **Metrics**: Use F1/Precision/Recall, not accuracy