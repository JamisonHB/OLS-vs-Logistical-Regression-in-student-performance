from ucimlrepo import fetch_ucirepo
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

def load_data():
    student = fetch_ucirepo(id=320)

    df = student.data.features.copy()
    targets = student.data.targets.copy()

    df["G3"] = targets["G3"]

    # Drop G1 and G2 to prevent data leakage
    df = df.drop(columns=["G1", "G2"], errors="ignore")

    return df

def preprocess_features(df):
    df = df.copy()

    y = df["G3"]
    X = df.drop(columns=["G3"])

    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    encoder = OneHotEncoder(sparse_output=False, drop="first")

    X_cat = encoder.fit_transform(X[categorical_cols])
    X_num = X[numerical_cols].values

    X_processed = np.hstack([X_num, X_cat])

    return X_processed, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_pass_fail(y):
    return (y >= 10).astype(int)

def standardize(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test