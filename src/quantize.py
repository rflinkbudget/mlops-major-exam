import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
from src.utils import load_data, dequantize, predict_with_weights  # ✅ imported from utils

# Load model
model = joblib.load("model.pth")

# Extract weights
coef = model.coef_
intercept = model.intercept_

# Save original parameters
joblib.dump((coef, intercept), "unquant_params.joblib")

# Quantize
coef_q = np.clip((coef * 10).astype(np.uint8), 0, 255)
intercept_q = np.clip(int(intercept * 10), 0, 255)
joblib.dump((coef_q, intercept_q), "quant_params.joblib")

# Dequantize
coef_dq, intercept_dq = dequantize(coef_q, intercept_q)

# Load test data
X, _ = load_data()
X_sample = X[:5]

# Predict
y_pred = predict_with_weights(X_sample, coef_dq, intercept_dq)
print("✅ Dequantized Predictions:", y_pred)

