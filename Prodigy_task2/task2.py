import pandas as pd
import matplotlib.pyplot as plt

# 1. READ DATASET

df = pd.read_csv("train.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# 2. DATA CLEANING

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Remove Cabin
df = df.drop(columns=["Cabin"])

# Create FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 3. SURVIVAL DISTRIBUTION

df["Survived"].value_counts().sort_index().plot(kind="bar")

plt.title("Titanic Survival Distribution")
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()

# 4. SURVIVAL BY GENDER

df.groupby("Sex")["Survived"].mean().mul(100).plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

plt.tight_layout()
plt.show()


# 5. SURVIVAL BY PASSENGER CLASS

df.groupby("Pclass")["Survived"].mean().mul(100).plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# 6. AGE DISTRIBUTION

plt.figure(figsize=(8, 5))

plt.hist(
    df["Age"],
    bins=20,
    edgecolor="black"
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()

# 7. AGE VS SURVIVAL

plt.figure(figsize=(9, 5))

plt.hist(
    df[df["Survived"] == 0]["Age"],
    bins=20,
    alpha=0.6,
    label="Did Not Survive"
)

plt.hist(
    df[df["Survived"] == 1]["Age"],
    bins=20,
    alpha=0.6,
    label="Survived"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.legend()

plt.tight_layout()
plt.show()

# 8. GENDER + CLASS

grouped = df.groupby(
    ["Pclass", "Sex"]
)["Survived"].mean().unstack()

grouped.plot(kind="bar")

plt.title("Survival Rate by Gender and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# 9. FAMILY SIZE

df.groupby(
    "FamilySize"
)["Survived"].mean().mul(100).plot(kind="bar")

plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# 10. CORRELATION

correlation = df[
    [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation)

# 11. CORRELATION VISUALIZATION

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=60
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# 12. FINAL RESULTS

print("\n==============================")
print("FINAL EDA RESULTS")
print("==============================")

print("\nOverall Survival Rate:")
print(df["Survived"].mean() * 100)

print("\nSurvival Rate by Gender:")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\nSurvival Rate by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean() * 100)

print("\nAverage Age:")
print(df["Age"].mean())

print("\nAverage Fare:")
print(df["Fare"].mean())
