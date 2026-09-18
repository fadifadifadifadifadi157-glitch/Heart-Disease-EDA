# ❤️ Heart Disease EDA

Exploratory Data Analysis (EDA) and basic preprocessing of a heart disease dataset using Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, and Scikit-learn.

This project focuses on understanding the structure and quality of the dataset, exploring relationships between patient attributes and the `HeartDisease` target, identifying unusual/invalid zero values, performing visual analysis, and applying basic statistical tests and preprocessing.

> **Note:** This is an educational data-analysis project. The relationships observed in this dataset should not be interpreted as medical diagnosis or clinical evidence.

---

## 📌 Project Overview

The project uses a heart disease dataset containing **918 rows and 12 columns**.

The notebook follows an EDA workflow:

1. Import the required Python libraries.
2. Load the dataset.
3. Inspect columns, shape, data types, and descriptive statistics.
4. Check duplicate and missing values.
5. Analyze the target variable.
6. Explore numerical features with histograms.
7. Identify and replace invalid zero values in `Cholesterol` and `RestingBP`.
8. Explore categorical features using count plots.
9. Compare numerical variables across the `HeartDisease` groups.
10. Examine numerical correlations with a heatmap.
11. Apply one-hot encoding to categorical variables.
12. Standardize selected numerical variables.
13. Use Chi-square tests for categorical features.
14. Use independent-samples t-tests for numerical features.

The current project is focused on **EDA and preprocessing**. No machine-learning model or prediction pipeline is included in the notebook.

---

## 📂 Project Structure

Based on the current project folder:

```text
Heart Disease EDA/
│
├── .ipynb_checkpoints/
│
├── heart.csv
│
├── Untitled.ipynb
|
└── README.md
```
## GitHub Repository:
https://github.com/fadifadifadifadifadi157-glitch/Heart-Disease-EDA.git

### Files and folders

| File / Folder | Description |
|---|---|
| `.ipynb_checkpoints/` | Jupyter Notebook's automatically generated checkpoint files |
| `heart.csv` | Heart disease dataset used by the notebook |
| `Untitled.ipynb` | Main Jupyter Notebook containing the EDA, cleaning, statistical analysis, and preprocessing |

### Dataset filename note

The notebook currently loads the dataset with:

```python
df = pd.read_csv('heart.csv')
```

Therefore, `heart.csv` should be present in the same directory as the notebook.

In the provided folder screenshot, the dataset is displayed as an Excel worksheet named `heart`. If your local file is actually an Excel file rather than CSV, either export it as `heart.csv` or change the loading code to the appropriate Pandas Excel reader.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Understand the structure of the heart disease dataset.
- Identify numerical and categorical variables.
- Check data types and descriptive statistics.
- Detect missing values and duplicate records.
- Investigate invalid zero values.
- Clean the affected numerical columns.
- Understand the distribution of the target variable.
- Explore categorical variables against `HeartDisease`.
- Compare numerical features between target groups.
- Study correlations between numerical variables.
- Perform statistical significance tests.
- Prepare categorical data using one-hot encoding.
- Standardize selected numerical features for possible downstream machine-learning use.

---

## 📊 Dataset Information

The dataset contains **918 observations** and **12 columns**.

### Columns

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Age of the individual |
| `Sex` | Categorical | Sex category (`M` / `F`) |
| `ChestPainType` | Categorical | Chest-pain category |
| `RestingBP` | Numerical | Resting blood pressure |
| `Cholesterol` | Numerical | Cholesterol measurement |
| `FastingBS` | Binary | Fasting blood sugar indicator |
| `RestingECG` | Categorical | Resting ECG category |
| `MaxHR` | Numerical | Maximum heart rate |
| `ExerciseAngina` | Categorical | Exercise-induced angina indicator |
| `Oldpeak` | Numerical | Oldpeak measurement |
| `ST_Slope` | Categorical | ST-segment slope category |
| `HeartDisease` | Binary target | Target variable (`0` / `1`) |

### Target variable

The target column is:

```text
HeartDisease
```

Target distribution in the dataset:

| HeartDisease | Count | Percentage |
|---:|---:|---:|
| `0` | 410 | 44.66% |
| `1` | 508 | 55.34% |
| **Total** | **918** | **100%** |

The target classes are not exactly equal, but neither class is extremely rare in this dataset.

---

## 🔎 Initial Data Inspection

The notebook uses several Pandas operations to inspect the dataset:

```python
df.columns
df.shape
df.info()
df.describe()
df.duplicated().sum()
df.isna().sum()
```

### Initial observations

- Dataset size: **918 × 12**
- Duplicate rows found: **0**
- Missing values: **0** in all columns
- Numerical and categorical variables are both present.
- `HeartDisease` is stored as an integer binary target.

---

## 🧹 Data Cleaning

### 1. Missing values

The notebook checks for missing values using:

```python
df.isna().sum()
```

No missing values were found.

---

### 2. Duplicate records

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

Result:

```text
0
```

Therefore, no duplicate rows were removed.

---

### 3. Invalid `Cholesterol = 0` values

The notebook identifies zero values in the `Cholesterol` column:

```python
df['Cholesterol'].value_counts()
```

There are **172 zero values**.

The notebook treats these zeros as invalid/missing-like values and replaces them with the mean of the non-zero cholesterol observations.

The calculated non-zero mean is approximately:

```text
244.64
```

Cleaning step:

```python
ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()

df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
```

After cleaning:

```text
Cholesterol zero values = 0
```

---

### 4. Invalid `RestingBP = 0` value

The notebook also identifies a zero value in `RestingBP`.

It calculates the mean of the non-zero observations:

```text
132.54
```

Then replaces the zero with this mean:

```python
Rbp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()

df['RestingBP'] = df['RestingBP'].replace(0, Rbp_mean).round(2)
```

After cleaning:

```text
RestingBP zero values = 0
```

### Cleaning summary

| Feature | Invalid zero values | Replacement |
|---|---:|---|
| `Cholesterol` | 172 | Mean of non-zero values ≈ 244.64 |
| `RestingBP` | 1 | Mean of non-zero values ≈ 132.54 |

---

## 📈 Exploratory Data Analysis

### Numerical Features

The notebook examines the distributions of:

- `Age`
- `RestingBP`
- `Cholesterol`
- `MaxHR`

using Seaborn histograms with KDE:

```python
sns.histplot(df[var], kde=True)
```

These plots are used to understand the distribution, spread, concentration, and unusual values in the numerical variables.

---

### Categorical Features

Count plots are used to explore:

- `Sex`
- `ChestPainType`
- `FastingBS`

with `HeartDisease` used as the hue.

Example:

```python
sns.countplot(x=df['Sex'], hue=df['HeartDisease'])
```

Additional categorical variables are included in the later statistical analysis:

- `RestingECG`
- `ExerciseAngina`
- `ST_Slope`

---

## 📦 Numerical vs Target Analysis

### Box Plot

The notebook uses a box plot to compare cholesterol distributions across the target groups:

```python
sns.boxplot(
    x='HeartDisease',
    y='Cholesterol',
    data=df
)
```

This helps visualize differences in:

- Median
- Interquartile range
- Spread
- Potential outliers

between `HeartDisease = 0` and `HeartDisease = 1`.

---

### Violin Plot

A violin plot is used for `Age`:

```python
sns.violinplot(
    x='HeartDisease',
    y='Age',
    data=df
)
```

The violin plot provides information about the distribution and density of ages within the two target groups.

---

## 🔥 Correlation Analysis

The notebook creates a correlation heatmap for numerical variables:

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
```

After the zero-value cleaning, the correlations with `HeartDisease` are approximately:

| Feature | Correlation with `HeartDisease` |
|---|---:|
| `Oldpeak` | `0.404` |
| `Age` | `0.282` |
| `FastingBS` | `0.267` |
| `RestingBP` | `0.118` |
| `Cholesterol` | `0.094` |
| `MaxHR` | `-0.400` |

These are **linear correlation measurements**, not proof of causation.

The strongest absolute numerical correlations with the target in this analysis are observed for `Oldpeak` and `MaxHR`.

---

## 🧪 Statistical Analysis

The notebook goes beyond visual EDA and performs statistical hypothesis tests.

Two types of tests are used:

1. Chi-square test for categorical variables.
2. Independent-samples t-test for numerical variables.

A significance level of:

```text
α = 0.05
```

is used.

---

## 1. Chi-square Test

The Chi-square test is applied to categorical features:

```python
categorical_cols = [
    'Sex',
    'ChestPainType',
    'FastingBS',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]
```

### Results

| Feature | Chi-square statistic | P-value | Notebook decision |
|---|---:|---:|---|
| `Sex` | 84.145 | 4.60e-20 | Reject Null |
| `ChestPainType` | 268.067 | 8.08e-58 | Reject Null |
| `FastingBS` | 64.321 | 1.06e-15 | Reject Null |
| `RestingECG` | 10.931 | 0.00423 | Reject Null |
| `ExerciseAngina` | 222.259 | 2.91e-50 | Reject Null |
| `ST_Slope` | 355.918 | 5.17e-78 | Reject Null |

At the 0.05 significance level, all six tested categorical variables show a statistically significant association with `HeartDisease` in this dataset.

A significant Chi-square result indicates an association between the categorical variable and target; it does **not** establish causation.

---

## 2. Independent-Samples T-Test

The notebook applies an independent-samples t-test to:

- `Age`
- `RestingBP`
- `Cholesterol`
- `MaxHR`
- `Oldpeak`

The two groups are:

```text
HeartDisease = 0
HeartDisease = 1
```

### Results

| Feature | T-statistic | P-value | Notebook decision |
|---|---:|---:|---|
| `Age` | -8.897 | 3.01e-18 | Reject Null |
| `RestingBP` | -3.595 | 3.42e-04 | Reject Null |
| `Cholesterol` | -2.860 | 0.00433 | Reject Null |
| `MaxHR` | 13.225 | 1.14e-36 | Reject Null |
| `Oldpeak` | -13.365 | 2.39e-37 | Reject Null |

At the 0.05 significance level, all five tested numerical variables show statistically significant differences in their group means in this dataset.

Approximate group means:

| Feature | HeartDisease = 0 | HeartDisease = 1 |
|---|---:|---:|
| `Age` | 50.55 | 55.90 |
| `RestingBP` | 130.18 | 134.45 |
| `Cholesterol` | 239.06 | 249.14 |
| `MaxHR` | 148.15 | 127.66 |
| `Oldpeak` | 0.41 | 1.27 |

These results describe differences within this dataset and should not be interpreted as clinical conclusions.

---

## 🔤 Categorical Encoding

The notebook converts categorical variables into numerical columns using one-hot encoding:

```python
df_encode = pd.get_dummies(
    df,
    drop_first=True
)
```

Boolean columns are then converted to integers:

```python
df_encode = df_encode.astype(int)
```

After encoding, the dataset contains:

```text
918 rows × 16 columns
```

Encoded categorical columns include:

```text
Sex_M
ChestPainType_ATA
ChestPainType_NAP
ChestPainType_TA
RestingECG_Normal
RestingECG_ST
ExerciseAngina_Y
ST_Slope_Flat
ST_Slope_Up
```

`drop_first=True` removes one reference category from each categorical variable to avoid redundant dummy columns.

---

## 📏 Standard Scaling

The notebook standardizes these numerical features:

```python
numerical_cols = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]
```

using:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_encode[numerical_cols] = scaler.fit_transform(
    df_encode[numerical_cols]
)
```

This transforms the selected numerical features to a standardized scale.

### Important note

The scaler is fitted directly on the complete dataset in this EDA notebook. If this preprocessing is later used for machine learning, the scaler should instead be fitted **only on the training data** and then used to transform validation/test data to avoid data leakage.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Statistical Analysis

- SciPy

### Machine Learning / Preprocessing

- Scikit-learn

### Additional Analysis Tool

The notebook also installs and uses:

```python
sheryanalysis
```

with:

```python
pip install sheryanalysis==0.1.0
```

and:

```python
import sheryanalysis as sh
sh.analyze(df)
```

---

## 📦 Main Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

Statistical testing:

```python
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
```

Preprocessing:

```python
from sklearn.preprocessing import StandardScaler
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Move into the project directory

```bash
cd "Heart Disease EDA"
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn sheryanalysis==0.1.0 jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
Untitled.ipynb
```

Make sure:

```text
heart.csv
```

is located in the same directory.

### 6. Run the notebook

Run the cells from top to bottom.

---

## 🔄 Project Workflow

```text
Heart Disease Dataset
        │
        ▼
Load Dataset
        │
        ▼
Initial Inspection
        │
        ├── Shape
        ├── Columns
        ├── Data Types
        ├── Statistics
        ├── Missing Values
        └── Duplicates
        │
        ▼
Data Quality Check
        │
        ├── Cholesterol = 0
        └── RestingBP = 0
        │
        ▼
Data Cleaning
        │
        ├── Replace invalid Cholesterol zeros
        └── Replace invalid RestingBP zero
        │
        ▼
Exploratory Data Analysis
        │
        ├── Histograms
        ├── Count Plots
        ├── Box Plot
        ├── Violin Plot
        └── Correlation Heatmap
        │
        ▼
Statistical Analysis
        │
        ├── Chi-square Tests
        └── Independent T-tests
        │
        ▼
Preprocessing
        │
        ├── One-Hot Encoding
        └── Standard Scaling
        │
        ▼
EDA / Preprocessing Completed
```

---

## 📌 Key Findings

Based on the analysis performed in the notebook:

1. The dataset contains **918 records and 12 original features**.
2. There are **no duplicate rows**.
3. There are **no Pandas-detected missing values**.
4. The dataset contains **172 zero values in `Cholesterol`**, which the notebook treats as invalid/missing-like values and replaces with the non-zero mean.
5. One zero value was found in `RestingBP` and replaced with the mean of the non-zero observations.
6. The target contains **508 records with `HeartDisease = 1`** and **410 with `HeartDisease = 0`**.
7. The numerical correlation analysis shows the largest absolute target correlations for `Oldpeak` and `MaxHR`.
8. All six categorical variables tested with Chi-square show statistically significant associations with the target at α = 0.05.
9. All five numerical variables tested with independent-samples t-tests show statistically significant differences between the two target groups at α = 0.05.
10. One-hot encoding expands the dataset from **12 columns to 16 columns**.
11. Five numerical features are standardized using `StandardScaler`.

---

## ⚠️ Important Limitations

This project is an exploratory analysis, so the findings have several limitations:

- Statistical association does not mean causation.
- The dataset alone cannot establish medical or clinical conclusions.
- Replacing zero values with the mean is a simple imputation strategy and may not be the only appropriate approach.
- The notebook does not document the original source/provenance of the dataset.
- The statistical tests are performed without a more extensive assumption-checking workflow.
- The current project does not include a train/test split or machine-learning evaluation.
- Standard scaling is performed on the complete dataset in the notebook; this is acceptable for demonstrating preprocessing, but should be changed when building a predictive model.
- No prediction, diagnosis, or clinical decision should be made from this EDA project.

---

## 🚀 Possible Future Improvements

The project can be extended with:

### Machine Learning

- Train/test split
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Gradient Boosting
- Hyperparameter tuning
- Cross-validation

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- ROC curve

### Better Preprocessing

- Use a reproducible preprocessing pipeline.
- Fit the scaler only on training data.
- Handle categorical variables through a `ColumnTransformer`.
- Compare different imputation strategies.
- Investigate outliers more systematically.

### Deployment

After developing and validating a machine-learning model, the project could be extended with:

- Streamlit frontend
- Saved preprocessing pipeline
- Saved trained model
- User input form
- Prediction interface

---

## 📚 Learning Outcomes

This project demonstrates practical experience with:

- Loading datasets with Pandas
- Understanding DataFrame structure
- Data cleaning
- Missing-value investigation
- Invalid-value detection
- Descriptive statistics
- Numerical feature distributions
- Categorical feature analysis
- Seaborn visualizations
- Correlation analysis
- Hypothesis testing
- Chi-square testing
- Independent-samples t-testing
- One-hot encoding
- Feature scaling
- Basic data-preprocessing workflow

---

## 👤 Author

**Fadi**

This project was created as part of a practical data-analysis and machine-learning learning journey.

---
