import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import load_data, dequantize, predict_with_weights

# Load model and data
model = joblib.load("model.pth")
X, y = load_data()
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_sample = X_test[:5]

# Extract weights
coef = model.coef_
intercept = model.intercept_

# -------------------------------------
# ❌ UINT8 Quantization
coef_q_8 = np.clip((coef * 10).astype(np.uint8), 0, 255)
intercept_q_8 = np.clip(int(intercept * 10), 0, 255)

coef_dq_8, intercept_dq_8 = dequantize(coef_q_8, intercept_q_8, scale=10)
pred_8bit = predict_with_weights(X_sample, coef_dq_8, intercept_dq_8)

# -------------------------------------
# ✅ INT16 Quantization
scale_factor = 1000
coef_q_16 = np.round(coef * scale_factor).astype(np.int16)
intercept_q_16 = int(round(intercept * scale_factor))

joblib.dump((coef_q_16, intercept_q_16), "quant_params.joblib")
coef_dq_16, intercept_dq_16 = dequantize(coef_q_16, intercept_q_16, scale=scale_factor)
pred_16bit = predict_with_weights(X_sample, coef_dq_16, intercept_dq_16)

# -------------------------------------
# Original model prediction
pred_original = model.predict(X_sample)

# -------------------------------------
# Print comparison table
print("\n📊 Comparison Table")
print(f"{'Index':<5} {'Original':>10} {'8-bit (uint8)':>18} {'16-bit (int16)':>18}")
print("-" * 55)
for i in range(5):
    print(f"{i:<5} {pred_original[i]:>10.4f} {pred_8bit[i]:>18.4f} {pred_16bit[i]:>18.4f}")
