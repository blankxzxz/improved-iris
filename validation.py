
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load the saved tuned model
model = joblib.load("iris_tuned_lr.pkl")

# Load the external dataset
external = pd.read_csv("dataset/testSet.csv")
original = pd.read_csv("dataset/iris.csv")

features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

# Validate required columns
missing = [col for col in features if col not in external.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Check measurements are numeric and complete
external[features] = external[features].apply(
    pd.to_numeric, errors="coerce"
)

if external[features].isna().any().any():
    raise ValueError("Dataset contains missing or non-numeric measurements.")

if (external[features] <= 0).any().any():
    raise ValueError("All flower measurements must be positive.")

# Flag exact feature matches with the original dataset
original_features = original[features].drop_duplicates()

external["_matches_original"] = external[features].apply(
    tuple, axis=1
).isin(
    original_features.apply(tuple, axis=1)
)

# Predict every flower
predictions = model.predict(external[features])

results = external.copy()
results["PredictedSpecies"] = predictions

probabilities = model.predict_proba(external[features])
results["ModelProbability"] = probabilities.max(axis=1).round(4)

print("=== EXTERNAL DATASET PREDICTIONS ===")
print(f"Total flowers: {len(results)}")
print(f"Exact matches to original measurements: "
      f"{results['_matches_original'].sum()}")

print("\nPredictions:")
print(results.to_string(index=False))

# Save all predictions
results.to_csv("prediction_results.csv", index=False)
print("\nSaved results to prediction_results.csv")

# Evaluate only if actual labels are provided
if "Species" in external.columns:
    actual = external["Species"].astype(str).str.strip()
    predicted = pd.Series(predictions, index=external.index)

    # Evaluate the complete labeled external dataset
    print("\n=== LABELED DATASET EVALUATION ===")
    print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(actual, predicted, labels=model.classes_))
    print("\nClassification report:")
    print(classification_report(
        actual,
        predicted,
        labels=model.classes_,
        zero_division=0
    ))

    # Evaluate only rows whose measurements don't match
    # any original flower's four feature values
    new_only = ~results["_matches_original"]

    if new_only.any():
        print("\n=== NON-MATCHING MEASUREMENTS ===")
        print(f"Rows: {new_only.sum()}")
        print(
            "Accuracy:",
            f"{accuracy_score(actual[new_only], predicted[new_only]):.4f}"
        )
    else:
        print("\nNo non-matching measurements available for separate evaluation.")
else:
    print("\nNo Species column found.")
    print("Predictions were generated, but accuracy cannot be calculated.")
