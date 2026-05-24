
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# 2. LOAD DATASET
# =========================

train_df = pd.read_csv(
    'data/loan_data.csv'
)

print("\nDataset Loaded Successfully")

print(train_df.head())

# =========================
# 3. DATA CLEANING
# =========================

print("\nMissing Values Before Cleaning:")

print(train_df.isnull().sum())

# Fill categorical missing values

train_df['Gender'] = (
    train_df['Gender'].fillna(
        train_df['Gender'].mode()[0]
    )
)

train_df['Married'] = (
    train_df['Married'].fillna(
        train_df['Married'].mode()[0]
    )
)

train_df['Dependents'] = (
    train_df['Dependents'].fillna(
        train_df['Dependents'].mode()[0]
    )
)

train_df['Self_Employed'] = (
    train_df['Self_Employed'].fillna(
        train_df['Self_Employed'].mode()[0]
    )
)

train_df['Credit_History'] = (
    train_df['Credit_History'].fillna(
        train_df['Credit_History'].mode()[0]
    )
)

# Fill numerical missing values

train_df['LoanAmount'] = (
    train_df['LoanAmount'].fillna(
        train_df['LoanAmount'].median()
    )
)

train_df['Loan_Amount_Term'] = (
    train_df['Loan_Amount_Term'].fillna(
        train_df['Loan_Amount_Term'].median()
    )
)

# Remove duplicates

train_df.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")

print(train_df.isnull().sum())

# =========================
# 4. EXPLORATORY DATA ANALYSIS
# =========================

# Loan Status Distribution

plt.figure(figsize=(6,4))

sns.countplot(
    x='Loan_Status',
    data=train_df
)

plt.title('Loan Status Distribution')

plt.show()

# Applicant Income Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    train_df['ApplicantIncome'],
    kde=True
)

plt.title('Applicant Income Distribution')

plt.show()

# Correlation Heatmap

numeric_df = train_df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Heatmap')

plt.show()

# =========================
# 5. FEATURE ENGINEERING
# =========================

# Convert Dependents safely

train_df['Dependents'] = (
    train_df['Dependents']
    .astype(str)
)

train_df['Dependents'] = (
    train_df['Dependents']
    .replace({
        '3+': '3',
        'nan': '0',
        'None': '0',
        '': '0'
    })
)

train_df['Dependents'] = pd.to_numeric(
    train_df['Dependents'],
    errors='coerce'
)

train_df['Dependents'] = (
    train_df['Dependents']
    .fillna(0)
)

train_df['Dependents'] = (
    train_df['Dependents']
    .astype(int)
)

# Total Income Feature

train_df['TotalIncome'] = (
    train_df['ApplicantIncome'] +
    train_df['CoapplicantIncome']
)

# EMI Feature

train_df['EMI'] = (
    train_df['LoanAmount'] /
    train_df['Loan_Amount_Term']
)

# Income Per Person Feature

train_df['Income_Per_Person'] = (
    train_df['ApplicantIncome'] /
    (train_df['Dependents'] + 1)
)

print("\nFeature Engineering Completed")

# =========================
# 6. ENCODING
# =========================

encoder = LabelEncoder()

categorical_columns = (
    train_df.select_dtypes(
        include='object'
    ).columns
)

for col in categorical_columns:

    train_df[col] = encoder.fit_transform(
        train_df[col].astype(str)
    )

print("\nEncoding Completed")

# =========================
# 7. SPLIT FEATURES & TARGET
# =========================

X = train_df.drop(
    'Loan_Status',
    axis=1
)

y = train_df['Loan_Status']

# =========================
# 8. HANDLE REMAINING NaN
# =========================

# Fill numerical columns

numeric_cols = X.select_dtypes(
    include=['int64', 'float64']
).columns

for col in numeric_cols:

    X[col] = X[col].fillna(
        X[col].median()
    )

# Fill categorical columns

categorical_cols = X.select_dtypes(
    include=['object']
).columns

for col in categorical_cols:

    X[col] = X[col].fillna(
        X[col].mode()[0]
    )

# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    0
)

# Final fill

X = X.fillna(0)

print("\nRemaining Missing Values:")

print(X.isnull().sum())

# =========================
# 9. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")

# =========================
# 10. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)

print("Test Shape:", X_test.shape)

# =========================
# 11. TRAIN MODELS
# =========================

models = {

    'Logistic Regression':
        LogisticRegression(),

    'Decision Tree':
        DecisionTreeClassifier(),

    'Random Forest':
        RandomForestClassifier(),

    'XGBoost':
        XGBClassifier(),

    'SVM':
        SVC(probability=True),

    'KNN':
        KNeighborsClassifier(),

    'Naive Bayes':
        GaussianNB()
}

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    trained_models[name] = model

print("\nAll Models Trained Successfully")

# =========================
# 12. MODEL EVALUATION
# =========================

best_accuracy = 0
best_model = None

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"\n{name}")

    print("Accuracy:", accuracy)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

print("\nBest Accuracy:", best_accuracy)

# =========================
# 13. CROSS VALIDATION
# =========================

cv_scores = cross_val_score(
    best_model,
    X_scaled,
    y,
    cv=5
)

print("\nCross Validation Scores:")

print(cv_scores)

print(
    "\nAverage CV Score:",
    cv_scores.mean()
)

# =========================
# 14. HYPERPARAMETER TUNING
# =========================

params = {

    'n_estimators': [100, 200],

    'max_depth': [5, 10]
}

grid = GridSearchCV(

    RandomForestClassifier(),

    params,

    cv=5,

    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")

print(grid.best_params_)

best_rf = grid.best_estimator_

# =========================
# 15. SAVE MODEL
# =========================

joblib.dump(
    best_rf,
    'best_model.pkl'
)

print("\nModel Saved Successfully")

# =========================
# 16. LOAD MODEL
# =========================

loaded_model = joblib.load(
    'best_model.pkl'
)

print("\nModel Loaded Successfully")

# =========================
# 17. FINAL OUTPUT
# =========================

print("\n=================================")

print("PROJECT COMPLETED SUCCESSFULLY")

print("=================================")

print("\nGenerated Files:")

print("1. best_model.pkl")

print("\nAlgorithms Used:")

print("- Logistic Regression")
print("- Decision Tree")
print("- Random Forest")
print("- XGBoost")
print("- SVM")
print("- KNN")
print("- Naive Bayes")

print("\nTechniques Used:")

print("- Data Cleaning")
print("- EDA")
print("- Feature Engineering")
print("- Encoding")
print("- Scaling")
print("- Cross Validation")
print("- Hyperparameter Tuning")
print("- Model Evaluation")