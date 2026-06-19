import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column):

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Missing value imputation
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
