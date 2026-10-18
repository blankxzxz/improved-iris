# Iris Species Classification — Experiment Log

## 1. Objective

Build a machine learning classifier that predicts the species of an Iris flower from its sepal and petal measurements.

## 2. Dataset and features

- **Dataset:** `dataset/iris.csv`
- **Input features:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species
- **Dataset size:** 150 observations
- **Evaluation split:** 80% training, 20% testing
- **Random state:** 42
- **Stratification:** Enabled
- **Cross-validation:** 5-fold

## 3. Baseline model comparison

| Model | Mean CV accuracy | CV standard deviation |
|---|---:|---:|
| Logistic Regression | 95.83% | 2.64 percentage points |
| Decision Tree | 94.17% | 2.04 percentage points |
| Random Forest | 95.00% | 1.67 percentage points |
| K-Nearest Neighbors | 96.67% | 3.12 percentage points |

The baseline models were evaluated using five-fold cross-validation on the training set. Logistic Regression and KNN used standardized features through a pipeline.

## 4. Hyperparameter tuning

### K-Nearest Neighbors

- **Method:** GridSearchCV
- **Scoring metric:** Accuracy
- **Cross-validation folds:** 5
- **Search space:** 5 neighbor counts × 2 weighting options × 2 distance metrics
- **Configurations evaluated:** 20

Selected configuration:
- `n_neighbors`: 5
- `weights`: `uniform`
- `metric`: `euclidean`

Best mean CV accuracy: **96.67%**

### Logistic Regression

- **Method:** GridSearchCV
- **Scoring metric:** Accuracy
- **Cross-validation folds:** 5
- **Configurations evaluated:** 15

Selected configuration:
- `C`: 10
- `max_iter`: 200
- `solver`: `lbfgs`

Best mean CV accuracy: **96.67%**

## 5. Initial test-set results

| Model | Test accuracy |
|---|---:|
| Baseline Logistic Regression | 93.33% |
| Tuned KNN | 93.33% |
| Tuned Logistic Regression | 100.00% |

The test set contains 30 observations. The tuned Logistic Regression model correctly classified all 30 observations in this evaluation.

## 6. Initial interpretation

Tuned Logistic Regression achieved the highest observed test accuracy among the three results recorded above. Tuned KNN and tuned Logistic Regression achieved the same mean cross-validation accuracy.

The results do not establish that Logistic Regression will always outperform KNN or achieve perfect accuracy on new data. The dataset and test sample are small.

## 7. Evaluation caveat

The test results were examined while comparing and selecting models. Therefore, the test set is no longer a completely independent final evaluation for the overall model-selection process.

Treat these results as an educational experiment. For a stronger final performance estimate, use an evaluation procedure that accounts for model selection, such as nested cross-validation, or evaluate a finalized model on a genuinely untouched dataset.

## 8. Current status

- [x] Train baseline models
- [x] Compare models using cross-validation
- [x] Tune KNN with GridSearchCV
- [x] Tune Logistic Regression with GridSearchCV
- [x] Evaluate selected models on the existing test split
- [ ] Verify and consolidate all evaluation results
- [ ] Save the selected preprocessing-and-model pipeline
- [ ] Add prediction and automated testing scripts
- [ ] Finalize README and GitHub documentation
