import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#load data
dataset = pd.read_csv("dataset/iris.csv")

x = dataset[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
]

y = dataset["Species"]

#split
x_train,x_test,y_train,y_test = train_test_split (
    x, y, test_size= 0.2 ,  random_state= 42,  stratify= y
)

#load models
LR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
DTC = DecisionTreeClassifier(random_state=42)
RFC = RandomForestClassifier(random_state=42)
KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

#crossval
cv1 = cross_val_score (
    LR,
    x_train, y_train,
    cv=5
)
cv2 = cross_val_score (
    DTC,
    x_train, y_train,
    cv=5
)
cv3 = cross_val_score (
    RFC,
    x_train, y_train,
    cv=5
)
cv4 = cross_val_score(
    KNN,
    x_train, y_train, 
    cv=5)

print("Logistic Regression:", cv1)
print("Decision Tree:", cv2)
print("Random Forest:", cv3)
print("KNN:", cv4)

print("LR Mean:", cv1.mean())
print("DT Mean:", cv2.mean())
print("RF Mean:", cv3.mean())
print("KNN Mean:", cv4.mean())

#fit
LR.fit(x_train, y_train)
DTC.fit(x_train, y_train)
RFC.fit(x_train, y_train)
KNN.fit(x_train, y_train)

print("LR accuracy:", LR.score(x_train, y_train))
print("DTC accuracy:", DTC.score(x_train, y_train))
print("RFC accuracy:", RFC.score(x_train, y_train))
print("KNN accuracy:", KNN.score(x_train, y_train))

print("LR Std:", cv1.std())
print("DT Std:", cv2.std())
print("RF Std:", cv3.std())
print("KNN Std:", cv4.std())

#testing other neighbor
for k in [1, 3, 5, 7, 9, 11]:
    KNN = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    scores = cross_val_score(
        KNN,
        x_train,
        y_train,
        cv=5
    )

    print(f"K={k} | Mean={scores.mean():.4f} | Std={scores.std():.4f}")

knn_pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

param_grid = {
    "kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11],
    "kneighborsclassifier__weights": ["uniform", "distance"],
    "kneighborsclassifier__metric": ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(x_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
best_knn = grid_search.best_estimator_

print("Test accuracy:", best_knn.score(x_test, y_test))