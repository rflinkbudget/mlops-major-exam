from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_data():
    data = fetch_california_housing()
    return data.data, data.target

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def dequantize(coef_q, intercept_q, scale=10.0):
    coef_dq = coef_q.astype(np.float32) / scale
    intercept_dq = intercept_q / scale
    return coef_dq, intercept_dq

def predict_with_weights(X, coef, intercept):
    return np.dot(X, coef) + intercept
