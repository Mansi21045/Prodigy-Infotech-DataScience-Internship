import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

# 1. READ DATASET

df = pd.read_csv(
    "twitter_training.csv",
    header=None
)

# Give columns names
df.columns = [
    "ID",
    "Entity",
    "Sentiment",
    "Tweet"
]

# 2. UNDERSTAND DATA

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# 3. CHECK MISSING VALUES

print("\nMissing Values:")
print(df.isnull().sum())

# 4. DATA CLEANING

# Remove rows with missing tweets
df = df.dropna(
    subset=["Tweet"]
)

# Remove duplicates
df = df.drop_duplicates()

# Convert Tweet to string
df["Tweet"] = df["Tweet"].astype(str)

# Convert text to lowercase
df["Tweet"] = df["Tweet"].str.lower()

# Remove extra spaces
df["Tweet"] = df["Tweet"].str.strip()

# 5. SENTIMENT CATEGORIES

print("\nSentiment Categories:")
print(
    df["Sentiment"].value_counts()
)

# 6. REMOVE IRRELEVANT

sentiment_df = df[
    df["Sentiment"].isin(
        [
            "Positive",
            "Negative",
            "Neutral"
        ]
    )
].copy()

# 7. OVERALL SENTIMENT

sentiment_counts = (
    sentiment_df["Sentiment"]
    .value_counts()
)

print("\nOverall Sentiment:")
print(sentiment_counts)

# 8. SENTIMENT BAR CHART

plt.figure(figsize=(8, 5))

sentiment_counts.plot(
    kind="bar"
)

plt.title(
    "Overall Sentiment Distribution"
)

plt.xlabel("Sentiment")

plt.ylabel(
    "Number of Tweets"
)

plt.tight_layout()

plt.show()

# 9. SENTIMENT PIE CHART

plt.figure(figsize=(7, 7))

sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title(
    "Overall Sentiment Distribution"
)

plt.ylabel("")

plt.tight_layout()

plt.show()

# 10. TOP ENTITIES

top_entities = (
    sentiment_df["Entity"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Entities:")
print(top_entities)

# 11. TOP ENTITY GRAPH

plt.figure(figsize=(10, 6))

top_entities.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 Entities by Number of Tweets"
)

plt.xlabel(
    "Number of Tweets"
)

plt.ylabel("Entity")

plt.tight_layout()

plt.show()

# 12. SENTIMENT BY ENTITY

top_entity_names = (
    top_entities.index
)

top_entity_data = sentiment_df[
    sentiment_df["Entity"].isin(
        top_entity_names
    )
]

entity_sentiment = pd.crosstab(
    top_entity_data["Entity"],
    top_entity_data["Sentiment"]
)


print("\nSentiment by Entity:")
print(entity_sentiment)

# 13. ENTITY SENTIMENT GRAPH

entity_sentiment.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Sentiment Distribution Across Top Entities"
)

plt.xlabel("Entity")

plt.ylabel(
    "Number of Tweets"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()

# 14. SENTIMENT PERCENTAGE

entity_percentage = (
    entity_sentiment
    .div(
        entity_sentiment.sum(axis=1),
        axis=0
    )
    * 100
)


entity_percentage.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title(
    "Sentiment Percentage by Entity"
)

plt.xlabel("Entity")

plt.ylabel(
    "Percentage"
)

plt.legend(
    title="Sentiment"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()

# 15. MOST POSITIVE ENTITIES

positive_percentage = (
    entity_percentage
    .get(
        "Positive",
        pd.Series()
    )
    .sort_values(
        ascending=False
    )
)

print(
    "\nTop Positive Entities:"
)

print(
    positive_percentage.head(10)
)

# 16. MOST NEGATIVE ENTITIES

negative_percentage = (
    entity_percentage
    .get(
        "Negative",
        pd.Series()
    )
    .sort_values(
        ascending=False
    )
)

print(
    "\nTop Negative Entities:"
)

print(
    negative_percentage.head(10)
)

# 17. WORD CLOUD

all_text = " ".join(
    sentiment_df["Tweet"]
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(
    all_text
)


plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Most Common Words in Tweets"
)

plt.show()

print(
    "\nTask 4 completed successfully!"
)
