import numpy as np
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def fetch_data_split():
    """Retrieve and split the California Housing dataset."""
    dataset = fetch_california_housing()
    features, targets = dataset.data, dataset.target
    return train_test_split(features, targets, test_size=0.2, random_state=42)


def build_regressor():
    """Instantiate a LinearRegression model."""
    return LinearRegression()


def persist_model(obj, outpath):
    """Persist model using joblib."""
    folder = os.path.dirname(outpath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(obj, outpath)


def retrieve_model(inpath):
    """Retrieve model using joblib."""
    return joblib.load(inpath)


def compute_scores(actual, predicted):
    """Compute R² and MSE metrics."""
    return r2_score(actual, predicted), mean_squared_error(actual, predicted)
    
def load_data():
    """Load California housing data (features and target)."""
    dataset = fetch_california_housing()
    return dataset.data, dataset.target    


def compress_to_uint8(arr):
    """Quantize float array to uint8 using per-element min-max scaling."""
    arr = np.asarray(arr)
    min_v = arr.copy()
    max_v = arr.copy()
    quant = np.zeros_like(arr, dtype=np.uint8)

    for i in range(arr.size):
        val = arr.flat[i]
        mn = val - 0.01  # Add buffer to min
        mx = val + 0.01  # Add buffer to max
        scale = 255 / (mx - mn) if mx != mn else 1.0
        quant.flat[i] = np.clip(np.round((val - mn) * scale), 0, 255).astype(np.uint8)
        min_v.flat[i] = mn
        max_v.flat[i] = mx

    return quant, min_v.astype(float), max_v.astype(float)
    
    
def decompress_from_uint8(quant, min_v, max_v):
    quant = np.asarray(quant, dtype=np.uint8)
    min_v = np.asarray(min_v)
    max_v = np.asarray(max_v)
    dequant = np.zeros_like(quant, dtype=np.float32)

    for i in range(quant.size):
        mn = min_v.flat[i]
        mx = max_v.flat[i]
        scale = (mx - mn) / 255 if mx != mn else 1.0
        dequant.flat[i] = quant.flat[i] * scale + mn

    return dequant

