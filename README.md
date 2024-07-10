
# Bank Marketing Data Analysis

This repository contains code for analyzing the bank marketing dataset. The analysis includes data loading, preprocessing, and various visualizations to understand the dataset better.

## Dataset

The dataset used in this analysis is the `bank_marketing_updated_v1.csv` file. It includes various attributes related to the bank's marketing campaigns.

## Dependencies

The following libraries are required to run the code:
- pandas
- numpy
- matplotlib
- seaborn

## Code Overview

### Loading the Dataset

The dataset is loaded using pandas:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv(r"C:\Users\patil\Downloads\bank_marketing_updated_v1.csv")
print(df.head())
```

### Data Preprocessing

1. **Checking the structure of the dataset:**
    ```python
    print(df.info())
    ```

2. **Imputing missing values:**
    - Age column with median:
        ```python
        median_age = df['age'].median()
        df['age'].fillna(median_age, inplace=True)
        ```
    - Month column with mode:
        ```python
        mode_month = df['month'].mode()[0]
        df['month'].fillna(mode_month, inplace=True)
        ```
    - Response column with mode:
        ```python
        mode_response = df['response'].mode()[0]
        df['response'].fillna(mode_response, inplace=True)
        ```

3. **Checking for null values:**
    ```python
    print(df.isnull().sum())
    ```

4. **Checking for duplicate rows:**
    ```python
    print(df.duplicated().sum())
    ```

5. **Dropping the `customerid` column:**
    ```python
    df.drop('customerid', axis='columns', inplace=True)
    ```

6. **Splitting the `jobedu` column:**
    ```python
    df['job'] = df['jobedu'].apply(lambda x: x.split(',')[0])
    df['education'] = df['jobedu'].apply(lambda x: x.split(',')[1])
    df.drop('jobedu', axis=1, inplace=True)
    ```

7. **Splitting the `month` column:**
    ```python
    df['month1'] = df['month'].apply(lambda x: x.split(',')[0])
    df['year'] = df['month'].apply(lambda x: x.split(',')[1])
    df.drop('month', axis=1, inplace=True)
    ```

### Univariate Analysis

1. **Categorical Unordered Analysis:**
    - Job status:
        ```python
        df.job.value_counts().plot.barh()
        plt.show()
        ```

    - Education variable:
        ```python
        df.education.value_counts(normalize=True).plot.pie()
        plt.show()
        ```

    - Month variable:
        ```python
        df.month1.value_counts().plot.barh()
        plt.show()
        ```

2. **Numerical Analysis:**
    - Age column:
        ```python
        df.age.describe()
        ```

### Bivariate Analysis

1. **Age vs Salary:**
    ```python
    df.plot.scatter(x="age", y="salary")
    plt.show()
    ```

2. **Age vs Balance:**
    ```python
    df.plot.scatter(x="age", y="balance")
    plt.show()
    ```

3. **Salary vs Balance:**
    ```python
    df.plot.scatter(x="salary", y="balance")
    plt.show()
    ```

4. **Correlation Heatmap:**
    ```python
    sns.heatmap(df[['age', 'salary', 'balance']].corr(), annot=True, cmap='Blues')
    plt.show()
    ```

### Categorical Variables Analysis

1. **Response Rate:**
    ```python
    df['response_rate'] = np.where(df.response == 'yes', 1, 0)
    df.response_rate.value_counts()
    ```

2. **Marital Status vs Response Rate:**
    ```python
    df.groupby('marital')['response_rate'].mean().plot.bar()
    plt.show()
    ```

3. **Temporal Trends by Month:**
    ```python
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='month1', order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    plt.title('Number of Observations by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    ```

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [Your Name] at [your-email@example.com].
```
