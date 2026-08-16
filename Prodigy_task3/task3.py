import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. READ DATASET

df = pd.read_csv("bank.csv", sep=";")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# 2. TARGET DISTRIBUTION

print("\nTarget distribution:")
print(df["y"].value_counts())

df["y"].value_counts().plot(kind="bar")

plt.title("Bank Marketing Subscription Distribution")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# 3. SEPARATE FEATURES AND TARGET

X = df.drop(columns=["y"])

y = df["y"].map({
    "no": 0,
    "yes": 1
})

# 4. IDENTIFY DATA TYPES

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# 5. ENCODE CATEGORICAL DATA

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# 6. CREATE DECISION TREE

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

# 7. CREATE PIPELINE

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# 8. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 9. TRAIN MODEL

pipeline.fit(
    X_train,
    y_train
)

# 10. MAKE PREDICTIONS

predictions = pipeline.predict(
    X_test
)

# 11. MODEL EVALUATION

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# 12. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["No", "Yes"]
)

plt.yticks(
    [0, 1],
    ["No", "Yes"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

# 13. CLASSIFICATION REPORT

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["No", "Yes"],
        zero_division=0
    )
)

# 14. DECISION TREE

feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(20, 10))

plot_tree(
    pipeline.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["No", "Yes"],
    rounded=True,
    fontsize=7,
    max_depth=3
)

plt.title(
    "Decision Tree Classifier"
)

plt.tight_layout()
plt.show()

# 15. FEATURE IMPORTANCE

importance = (
    pipeline
    .named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Important Features:")

print(
    importance_df.head(10)
)

# 16. FEATURE IMPORTANCE GRAPH

top_features = importance_df.head(10)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 10 Important Features"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

print("\nTask 3 completed successfully!")
