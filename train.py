
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
    
# Load data
dataset = pd.read_csv("dataset/iris.csv")

# Select features and target
x = dataset[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]
y = dataset["Species"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, "iris_model.pkl")

# Basic evaluation
print("Train accuracy:", model.score(x_train, y_train))
print("Test accuracy:", model.score(x_test, y_test))
