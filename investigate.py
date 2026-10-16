
import pandas as pd
import matplotlib.pyplot as plt

# Load data
dataset = pd.read_csv("dataset/iris.csv")

# Missing values
print("Missing values:")
print(dataset.isnull().sum())

# Potential duplicate measurements, ignoring Id
duplicates = dataset[
    dataset.drop(columns=["Id"]).duplicated(keep=False)
]
print("\nPotential duplicate measurements:")
print(duplicates.to_string(index=False))

# Descriptive statistics
print("\nDescriptive statistics:")
print(dataset.describe())

# Species distribution
print("\nSpecies distribution:")
print(dataset["Species"].value_counts())

# Visualize petal measurements
for species in dataset["Species"].unique():
    flowers = dataset[dataset["Species"] == species]
    plt.scatter(
        flowers["PetalLengthCm"],
        flowers["PetalWidthCm"],
        label=species
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Measurements by Species")
plt.legend()
# Highlight the misclassified flower
plt.scatter(
    5.0, 1.7,
    color="red",
    marker="x",
    s=150,
    label="Misclassified flower"
)

plt.legend()
plt.show()


