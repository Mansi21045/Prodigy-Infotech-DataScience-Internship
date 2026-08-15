# Prodigy Infotech - Data Science Task 2

## Task Objective

Perform data cleaning and Exploratory Data Analysis (EDA) on a dataset.
Explore the relationships between variables and identify patterns and trends
in the data.

## Dataset

For this task, I used the Titanic dataset.

The dataset contains information about Titanic passengers, including:

- Passenger ID
- Survival
- Passenger Class
- Name
- Gender
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Ticket
- Fare
- Cabin
- Port of Embarkation

## Technologies Used

- Python
- Pandas
- Matplotlib
- VS Code

## Data Cleaning

The following data-cleaning steps were performed:

1. Checked the dataset for missing values.
2. Checked and removed duplicate records.
3. Filled missing Age values using the median age.
4. Filled missing Embarked values using the most common value (mode).
5. Removed the Cabin column because it contained a large number of missing values.
6. Created a new `FamilySize` variable using the number of siblings/spouses,
   parents/children, and the passenger themselves.

## Exploratory Data Analysis

The following analyses were performed:

- Survival distribution
- Survival rate by gender
- Survival rate by passenger class
- Age distribution
- Age distribution by survival outcome
- Survival rate by gender and passenger class
- Survival rate by family size
- Correlation analysis of numerical variables

## Visualizations

The project includes visualizations such as:

- Survival Distribution
- Survival Rate by Gender
- Survival Rate by Passenger Class
- Age Distribution
- Age Distribution by Survival
- Gender and Passenger Class Analysis
- Family Size Analysis
- Correlation Matrix

## Key Findings

### 1. Gender and Survival

Female passengers had a significantly higher survival rate than male
passengers.

- Male survival rate: approximately 18.89%
- Female survival rate: approximately 74.20%

### 2. Passenger Class and Survival

Passenger class had a noticeable relationship with survival.

First-class passengers had the highest survival rate, while third-class
passengers had the lowest survival rate.

### 3. Age and Survival

The age distribution of survivors and non-survivors showed noticeable
differences. This indicates that age was also relevant to the survival outcome.

### 4. Gender and Passenger Class

When gender and passenger class were analyzed together, clear differences
in survival rates could be observed.

### 5. Family Size

Survival rates varied across different family sizes, showing that family
structure may also have influenced survival outcomes.

## Conclusion

The Titanic dataset was cleaned and analyzed using Python, Pandas, and
Matplotlib. Exploratory Data Analysis helped identify important relationships
between passenger survival and variables such as gender, passenger class,
age, and family size.

The visualizations made it easier to understand patterns and trends in the
dataset and demonstrated how EDA can be used to extract meaningful insights
from real-world data.

## Project Files

- `train.csv` - Original Titanic dataset
- `task2.py` - Python source code
- `Task2_Cleaned_Titanic.csv` - Cleaned dataset
- `Task2_Summary_Statistics.csv` - Summary statistics
- `Task2_01_Survival_Distribution.png` - Survival distribution
- `Task2_02_Survival_by_Gender.png` - Survival by gender
- `Task2_03_Survival_by_Class.png` - Survival by passenger class
- `Task2_04_Age_Distribution.png` - Age distribution
- `Task2_05_Age_by_Survival.png` - Age and survival analysis
- `Task2_06_Gender_Class_Analysis.png` - Gender and class analysis
- `Task2_07_Family_Size.png` - Family size analysis
- `Task2_08_Correlation_Matrix.png` - Correlation matrix
- `README.md` - Project documentation
