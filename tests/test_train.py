import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def test_data_loading():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0
    assert data.target.shape[0] > 0

def test_model_training():
    data = fetch_california_housing()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, 'coef_')

def test_model_saved():
    # Ensure model file exists after train.py has been run
    assert os.path.exists("model.pth")
    model = joblib.load("model.pth")
    assert isinstance(model, LinearRegression)
