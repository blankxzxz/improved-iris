
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Load data and recreate the same split
dataset = pd.read_csv("dataset/iris.csv")

x = dataset[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]
y = dataset["Species"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the saved model
model = joblib.load("iris_model.pkl")
LR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
# Predict
y_pred = model.predict(x_test)
y_pred_try = model.predict_proba(x_test)
prob_df = pd.DataFrame(
    y_pred_try,
    columns=model.classes_,
    index=x_test.index
)
# Detailed evaluation
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Cross-validation on training data
cv_scores = cross_val_score(
    LR,
    x_train, y_train,
    cv=5
)
print("\nCV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Report incorrect predictions
errors = x_test.copy()
errors["Actual"] = y_test
errors["Predicted"] = y_pred

errors = errors[errors["Actual"] != errors["Predicted"]]

print("\nMisclassified flowers:")
print(errors.to_string())

# Compare the misclassified flower with both species
for species in ["Iris-versicolor", "Iris-virginica"]:
    flowers = dataset[dataset["Species"] == species]

    print(f"\n{species} petal measurements:")
    print(flowers[["PetalLengthCm", "PetalWidthCm"]].describe())

# Get predicted probabilities
y_proba = model.predict_proba(x_test)

# Show probabilities for misclassified flowers
errors = x_test.copy()
errors["Actual"] = y_test
errors["Predicted"] = y_pred

errors["Confidence"] = y_proba.max(axis=1)

print("\nMisclassified flowers and confidence:")
print(errors[errors["Actual"] != errors["Predicted"]].to_string())

# Find the misclassified flower
wrong_idx = errors.index[errors["Actual"] != errors["Predicted"]][0]

# Calculate distances from it to every flower
distances = pairwise_distances(
    x.loc[[wrong_idx]],
    x
)[0]

nearest = dataset.copy()
nearest["Distance"] = distances
nearest = nearest[nearest.index != wrong_idx]

print(nearest.nsmallest(5, "Distance")[
    ["Species", "SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm", "Distance"]
].to_string(index=True))

# Show the model's probabilities for every test flower
probabilities = pd.DataFrame(
    y_proba,
    columns=model.classes_,
    index=x_test.index
)

print("\nPrediction probabilities:")
print(probabilities.round(3).to_string())
print(confusion_matrix(y_test, y_pred))
print(prob_df.round(3))
