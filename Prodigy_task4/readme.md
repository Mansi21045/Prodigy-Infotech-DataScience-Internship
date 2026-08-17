# Prodigy Infotech - Data Science Task 4

## Task Objective

Analyze and visualize sentiment patterns in social media data to understand
public opinion and attitudes towards specific topics or brands.

## Dataset

For this task, I used the Twitter Entity Sentiment Analysis dataset provided
for Task 4 by Prodigy Infotech.

The dataset contains Twitter/social media posts along with the entity or topic
associated with each post and its sentiment.

### Main Columns

- `ID` - Unique identifier of the record
- `Entity` - Topic, brand, or entity associated with the tweet
- `Sentiment` - Sentiment classification of the tweet
- `Tweet` - Text content of the tweet

### Sentiment Categories

The dataset contains the following sentiment categories:

- Positive
- Negative
- Neutral
- Irrelevant

For the main sentiment analysis, Positive, Negative, and Neutral tweets were
used.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- VS Code

## Data Cleaning

The following data-cleaning steps were performed:

1. Loaded the Twitter training dataset using Pandas.
2. Assigned meaningful column names to the dataset.
3. Checked the dataset for missing values.
4. Removed records with missing tweet text.
5. Removed duplicate records.
6. Converted tweet data into string format.
7. Converted tweet text to lowercase.
8. Removed unnecessary spaces from tweet text.
9. Filtered out irrelevant sentiment records for the main sentiment analysis.

## Exploratory Data Analysis

The following analyses were performed:

- Overall sentiment distribution
- Percentage distribution of sentiments
- Identification of the most frequently discussed entities
- Sentiment distribution across top entities
- Sentiment percentage by entity
- Identification of entities with higher positive sentiment
- Identification of entities with higher negative sentiment

## Visualizations

The project includes visualizations such as:

- Overall Sentiment Distribution
- Sentiment Percentage Distribution
- Top 10 Entities by Number of Tweets
- Sentiment Distribution Across Top Entities
- Sentiment Percentage by Entity

## Key Findings

### 1. Overall Sentiment

The dataset contains positive, negative, and neutral opinions about different
entities and topics.

The sentiment distribution helps provide an overview of the general public
attitude represented in the collected social media posts.

### 2. Popular Entities

Some entities appear more frequently than others, indicating that they were
more actively discussed in the dataset.

### 3. Sentiment Differences Between Entities

Different entities show different proportions of positive, negative, and
neutral opinions.

This demonstrates that public opinion can vary significantly depending on the
topic or brand being discussed.

### 4. Positive Sentiment

The percentage analysis helps identify entities that received a relatively
higher proportion of positive opinions.

### 5. Negative Sentiment

The analysis also identifies entities associated with a relatively higher
proportion of negative opinions.

## Conclusion

The Twitter Entity Sentiment dataset was analyzed using Python, Pandas,
Matplotlib, and Seaborn.

The analysis helped identify sentiment patterns and differences in public
opinion across various topics and entities. Visualizations made it easier to
compare positive, negative, and neutral sentiments and understand how public
attitudes vary across different entities.

This task provided practical experience in social media data analysis,
data cleaning, exploratory data analysis, and data visualization.

## Project Files

- `twitter_training.csv` - Original Twitter sentiment dataset
- `task4.py` - Python source code
- `README.md` - Project documentation
- `sentiment_distribution.png` - Overall sentiment visualization
- `sentiment_pie.png` - Sentiment percentage visualization
- `top_entities.png` - Top entities visualization
- `entity_sentiment.png` - Sentiment distribution across entities
- `sentiment_percentage.png` - Sentiment percentage by entity

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Sentiment Analysis
- Categorical Data Analysis
- Data Visualization
- Python Programming
- Pandas
- Matplotlib
- Seaborn
