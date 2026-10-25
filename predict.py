
import joblib

# Load the tuned model
model = joblib.load("iris_tuned_lr.pkl")

# Get flower measurements from the user
print("=== Iris Flower Predictor ===")

sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Make prediction
new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(new_flower)

print("\nPredicted species:", prediction[0])
