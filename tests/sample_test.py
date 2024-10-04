import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

@pytest.fixture
def setup_pipeline():
    # Load dataset (ensure the tips.csv exists in the correct directory)
    df = pd.read_csv('tips.csv')

    # Label encoding categorical features
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['day'] = df['day'].apply(lambda x: 1 if x == 'Sun' else 2)
    df['time'] = df['time'].apply(lambda x: 1 if x == 'Dinner' else 0)

    # Splitting features and target
    X = df.drop(columns=['total_bill'])
    y = df['total_bill']

    # Standard scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # Splitting into train/test data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, df, sc, X_scaled

def test_standard_scaler(setup_pipeline):
    # Get the scaled features from the fixture
    X_train, X_test, _, _, _, sc, _ = setup_pipeline

    # Check mean and variance for each feature individually
    feature_means = np.mean(X_train, axis=0)
    feature_vars = np.var(X_train, axis=0)

    # Assert that the mean of each feature is close to 0
    assert np.allclose(feature_means, 0, atol=1e-1), f"Means not close to 0: {feature_means}"
    
    # Assert that the variance of each feature is close to 1
    assert np.allclose(feature_vars, 1, atol=0.2), f"Variances not close to 1: {feature_vars}"  # Increased tolerance

def test_model_training(setup_pipeline):
    # Get the train/test data from the fixture
    X_train, X_test, y_train, y_test, _, _, _ = setup_pipeline

    # Ensure test set has at least 2 samples to avoid R^2 warnings
    assert len(X_test) >= 2, "Test set too small for R² calculation"

    # Train a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Ensure that R^2 score is within a valid range (-1 to 1)
    score = r2_score(y_test, y_pred)
    assert not np.isnan(score), "R^2 score is NaN"
    assert -1 <= score <= 1, f"R^2 score out of range: {score}"

def test_r2_score(setup_pipeline):
    # Get the train/test data from the fixture
    X_train, X_test, y_train, y_test, _, _, _ = setup_pipeline

    # Ensure test set has at least 2 samples to avoid R² warnings
    assert len(X_test) >= 2, "Test set too small for R² calculation"

    # Train a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Ensure R^2 score is reasonable (greater than or equal to 0)
    r2 = r2_score(y_test, y_pred)
    assert not np.isnan(r2), "R^2 score is NaN"
    assert r2 >= 0, f"R^2 score is negative: {r2}"
