import pandas as pd
import numpy as np


def create_features(df):

    # -----------------------------
    # Clean Dependents Column
    # -----------------------------

    # Convert to string first
    df['Dependents'] = df['Dependents'].astype(str)

    # Replace invalid values
    df['Dependents'] = df['Dependents'].replace({
        '3+': '3',
        'nan': '0',
        'None': '0',
        '': '0'
    })

    # Fill remaining missing values
    df['Dependents'] = df['Dependents'].fillna('0')

    # Convert safely to numeric
    df['Dependents'] = pd.to_numeric(
        df['Dependents'],
        errors='coerce'
    )

    # Replace any remaining NaN with 0
    df['Dependents'] = df['Dependents'].fillna(0)

    # Convert to integer
    df['Dependents'] = df['Dependents'].astype(int)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # Total Income
    df['TotalIncome'] = (
        df['ApplicantIncome'] +
        df['CoapplicantIncome']
    )

    # EMI Feature
    df['EMI'] = (
        df['LoanAmount'] /
        df['Loan_Amount_Term']
    )

    # Income Per Person
    df['Income_Per_Person'] = (
        df['ApplicantIncome'] /
        (df['Dependents'] + 1)
    )

    return df