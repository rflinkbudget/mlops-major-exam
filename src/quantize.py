import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load model
model = joblib.load("model.pth")

# Extract parameters
coef = model.coef_
intercept = model.intercept_

# Save raw parameters
joblib.dump((coef, intercept), "unquant_params.joblib")

# Manual quantization
coef_q = np.clip((coef * 10).astype(np.uint8), 0, 255)
intercept_q = np.clip(int(intercept * 10), 0, 255)

# Save quantized parameters
joblib.dump((coef_q, intercept_q), "quant_params.joblib")

# Dequantize
coef_dq = coef_q.astype(np.float32) / 10.0
intercept_dq = intercept_q / 10.0

# Predict with dequantized weights
def predict(X):
    return np.dot(X, coef_dq) + intercept_dq

# Load sample data for test
data = fetch_california_housing()
X_sample = data.data[:5]
y_pred = predict(X_sample)

print("✅ Dequantized Predictions:", y_pred)
