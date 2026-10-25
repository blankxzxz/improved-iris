# Iris Classification — Experiment Log

## 1. Objective
Compare baseline classification models and tune Logistic Regression and KNN to find a suitable Iris species classifier.

## 2. Dataset
- File: `dataset/iris.csv`
- Samples: 150
- Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- Target: `Species`

## 3. Experimental Setup
- Training/test split: 80% / 20%
- Random state: 42
- Stratification: Yes
- Cross-validation: 5-fold
- Scaling: `StandardScaler` inside pipelines for Logistic Regression and KNN
- Scoring metric: Accuracy

## 4. Baseline Cross-Validation Results
| Model | Mean CV Accuracy | CV Standard Deviation | Training Accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.9583 | 0.0264 | 0.9583 |
| Decision Tree | 0.9417 | 0.0204 | 1.0000 |
| Random Forest | 0.9500 | 0.0167 | 1.0000 |
| KNN (k=5) | 0.9667 | 0.0312 | 0.9750 |

## 5. Hyperparameter Tuning
### KNN
- Search parameters: `n_neighbors`, `weights`, `metric`
- Best CV accuracy: 0.9667
- Best parameters: Confirm from the latest script output.

### Logistic Regression
- Search parameters: `C`, `solver`, `max_iter`
- Best CV accuracy: 0.9667
- Best parameters: `C=10`, `solver=lbfgs`, `max_iter=200`

## 6. Test-Set Results
Record the latest output from a single run of `compare_models.py`:
- Tuned KNN accuracy: Confirm from output.
- Tuned Logistic Regression accuracy: Confirm from output.
- Confusion matrices and classification reports: Confirm from output.

## 7. Interpretation
- Cross-validation estimates performance across multiple training/validation folds.
- Test accuracy measures performance on one held-out test set.
- Training accuracy alone is not sufficient to judge generalization.
- Perfect test accuracy on this split does not guarantee perfect performance on unseen data.

## 8. Limitations
The test set has already been examined during experimentation. Its score should be treated as an exploratory result rather than a fully independent final estimate.

## 9. Reproducibility
- Main experiment script: `compare_models.py`
- Verification script: `evaluate_models.py`
- Dataset: `dataset/iris.csv`
- Random state: 42
- Python and scikit-learn versions: Record from the environment used.

## 10. Final Model
- Selected model: Pending final decision
- Saved artifact: Pending
- Prediction test: Pending

## 11. Software Environment

- Python version: `3.14.6`
- scikit-learn version: '1.9.0'


