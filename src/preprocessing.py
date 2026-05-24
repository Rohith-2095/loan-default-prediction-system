import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):

    # Fill missing numerical values
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    # Fill categorical missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def encode_features(df):

    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include='object').columns

    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def scale_features(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled