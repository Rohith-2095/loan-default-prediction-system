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

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv('data/loan_data.csv')

print("Dataset Loaded Successfully")

# ============================================
# DATA CLEANING
# ============================================

# Fill categorical missing values

categorical_columns = [
    'Gender',
    'Married',
    'Dependents',
    'Self_Employed',
    'Credit_History'
]

for col in categorical_columns:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

# Fill numerical missing values

numerical_columns = [
    'LoanAmount',
    'Loan_Amount_Term'
]

for col in numerical_columns:

    df[col] = df[col].fillna(
        df[col].median()
    )

# Remove duplicates

df.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning")

print(df.isnull().sum())

# ============================================
# FEATURE ENGINEERING
# ============================================

# Clean Dependents column

df['Dependents'] = (
    df['Dependents']
    .astype(str)
)

df['Dependents'] = (
    df['Dependents']
    .replace({
        '3+': '3',
        'nan': '0',
        'None': '0',
        '': '0'
    })
)

df['Dependents'] = pd.to_numeric(
    df['Dependents'],
    errors='coerce'
)

df['Dependents'] = (
    df['Dependents']
    .fillna(0)
)

df['Dependents'] = (
    df['Dependents']
    .astype(int)
)

# Create new features

df['TotalIncome'] = (
    df['ApplicantIncome'] +
    df['CoapplicantIncome']
)

df['EMI'] = (
    df['LoanAmount'] /
    df['Loan_Amount_Term']
)

df['Income_Per_Person'] = (
    df['ApplicantIncome'] /
    (df['Dependents'] + 1)
)

print("\nFeature Engineering Completed")

# ============================================
# ENCODING
# ============================================

encoder = LabelEncoder()

object_columns = df.select_dtypes(
    include='object'
).columns

for col in object_columns:

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

print("\nEncoding Completed")

# ============================================
# SPLIT FEATURES & TARGET
# ============================================

X = df.drop(
    'Loan_Status',
    axis=1
)

y = df['Loan_Status']

# ============================================
# REMOVE ALL NaN VALUES
# ============================================

# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill all remaining NaN

for col in X.columns:

    if X[col].dtype in ['int64', 'float64']:

        X[col] = X[col].fillna(
            X[col].median()
        )

    else:

        X[col] = X[col].fillna(
            X[col].mode()[0]
        )

# FINAL SAFETY CHECK

print("\nRemaining Missing Values")

print(X.isnull().sum())

# Replace any remaining NaN with 0

X = X.fillna(0)

# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# TRAIN MODELS
# ============================================

models = {

    'Logistic Regression':
        LogisticRegression(max_iter=1000),

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

# ============================================
# MODEL EVALUATION
# ============================================

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

# ============================================
# CROSS VALIDATION
# ============================================

scores = cross_val_score(
    best_model,
    X_scaled,
    y,
    cv=5
)

print("\nCross Validation Scores")

print(scores)

print("\nAverage CV Score")

print(scores.mean())

# ============================================
# HYPERPARAMETER TUNING
# ============================================

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

print("\nBest Parameters")

print(grid.best_params_)

best_rf = grid.best_estimator_

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(
    best_rf,
    'best_model.pkl'
)

print("\nModel Saved Successfully")

# ============================================
# FINAL OUTPUT
# ============================================

print("\n===================================")

print("PROJECT COMPLETED SUCCESSFULLY")

print("===================================")

print("\nGenerated File:")

print("best_model.pkl")