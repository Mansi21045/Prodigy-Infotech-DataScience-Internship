# Prodigy Infotech - Data Science Task 3

## Task Objective

Build a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

## Dataset

For this task, I used the **Bank Marketing Dataset** provided for the Prodigy Infotech Data Science Task 3.

The dataset contains information about customers and their interactions with a bank's marketing campaign.

The target variable is:

* `yes` - Customer subscribed to the term deposit
* `no` - Customer did not subscribe to the term deposit

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* VS Code

## Machine Learning Model

A **Decision Tree Classifier** was used for this task.

The Decision Tree learns patterns from customer demographic and behavioral information and predicts whether a customer is likely to subscribe to the bank's term deposit.

## Data Preparation

The following steps were performed:

1. Loaded the Bank Marketing dataset using Pandas.
2. Checked the structure and information of the dataset.
3. Checked for missing values.
4. Separated the input features from the target variable.
5. Converted the target variable:

   * `no` → `0`
   * `yes` → `1`
6. Identified categorical and numerical variables.
7. Applied One-Hot Encoding to categorical variables.
8. Split the dataset into training and testing sets using an 80:20 ratio.

## Model Configuration

The Decision Tree Classifier was configured with:

* Maximum depth: `5`
* Minimum samples per leaf: `10`
* Class weight: `balanced`
* Random state: `42`

These settings help keep the tree reasonably simple and reduce overfitting.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 75.47% |
| Precision | 30.33% |
| Recall    | 87.50% |
| F1 Score  | 45.05% |

The model achieved a relatively high recall, meaning it was able to identify a large proportion of customers who actually subscribed. Precision was lower, indicating that the model also produced a considerable number of false-positive predictions.

## Confusion Matrix

A confusion matrix was created to compare the actual customer outcomes with the model's predictions.

It helps identify:

* Correct `No` predictions
* Correct `Yes` predictions
* Incorrect `No` predictions
* Incorrect `Yes` predictions

## Feature Importance

Feature importance was analyzed to identify the variables that contributed most to the Decision Tree's predictions.

The most important features in the fitted model included:

1. `duration`
2. `contact`
3. `poutcome`
4. `month`
5. `age`
6. `day`
7. `balance`
8. `education`

The `duration` variable had the highest importance in the fitted model.

### Important Note

The `duration` variable represents the duration of the current marketing contact. Therefore, it is useful for understanding the outcome of an observed contact, but it would not be available before making a call. For a pre-call customer targeting model, this variable should be excluded.

## Conclusion

In this task, a Decision Tree Classifier was successfully developed to predict whether bank customers would subscribe to a term deposit.

The project provided practical experience in:

* Data preprocessing
* Categorical data encoding
* Train-test splitting
* Decision Tree classification
* Model prediction
* Model evaluation
* Confusion matrix analysis
* Feature importance analysis

The results demonstrate how machine learning can be used to analyze customer data and predict marketing outcomes.

## Project Files

* `bank.csv` - Bank Marketing dataset
* `task3.py` - Python source code
* `README.md` - Project documentation
* `decision_tree.png` - Decision Tree visualization
* `confusion_matrix.png` - Confusion Matrix visualization
* `feature_importance.png` - Feature Importance visualization
* `predictions.csv` - Model predictions
* `feature_importance.csv` - Feature importance values
* `model_report.txt` - Model evaluation report
