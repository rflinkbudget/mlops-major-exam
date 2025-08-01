import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib
from src import my_utils as utils
  # Importing entire utils module

def quantize_main():
    print("\n🔧 Loading trained model...")
    model = utils.retrieve_model("model.pth")

    # Extract weights
    coef = model.coef_
    intercept = model.intercept_
    print(f"Model coefficients: {coef.shape}, Intercept: {intercept:.6f}")

    # Save original unquantized parameters
    os.makedirs("models", exist_ok=True)
    joblib.dump({'coef': coef, 'intercept': intercept}, "models/unquant_params.joblib")

    # Quantize to 8-bit using min-max scaling
    q_coef, coef_min, coef_max = utils.compress_to_uint8(coef)
    q_intercept, int_min, int_max = utils.compress_to_uint8(np.array([intercept]))

    quant_params = {
        'quant_coef8': q_coef,
        'coef8_min': coef_min,
        'coef8_max': coef_max,
        'quant_intercept8': q_intercept[0],
        'intercept_min': int_min,
        'intercept_max': int_max
    }
    joblib.dump(quant_params, "models/quant_params.joblib", compress=3)

    # File size comparison
    size_orig = os.path.getsize("model.pth") / 1024
    size_quant = os.path.getsize("models/quant_params.joblib") / 1024
    print(f"\n📦 Original model size: {size_orig:.2f} KB")
    print(f"📦 Quantized model size: {size_quant:.2f} KB")

    # Dequantize for testing
    d_coef = utils.decompress_from_uint8(q_coef, coef_min, coef_max)
    d_intercept = utils.decompress_from_uint8(np.array([quant_params['quant_intercept8']]),
                                              np.array([int_min]),
                                              np.array([int_max]))[0]

    # Error check
    coef_error = np.abs(coef - d_coef).max()
    intercept_error = abs(intercept - d_intercept)
    print(f"\n⚠️  Max coef error: {coef_error:.6f}")
    print(f"⚠️  Intercept error: {intercept_error:.6f}")

    # Run inference on test data
    X_train, X_test, y_train, y_test = utils.fetch_data_split()
    preds_quant = X_test @ d_coef + d_intercept
    preds_orig = model.predict(X_test)

    diff = np.abs(preds_orig - preds_quant)
    print(f"\n📊 Inference difference (original vs quantized)")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    if diff.max() < 0.1:
        print("✅ Quantization quality: excellent")
    elif diff.max() < 1.0:
        print("✅ Quantization quality: acceptable")
    else:
        print("❌ Quantization quality: poor")

    # Performance metrics
    r2_quant, mse_quant = utils.compute_scores(y_test, preds_quant)
    print(f"\n📈 Quantized Model Metrics:")
    print(f"R² Score: {r2_quant:.4f}")
    print(f"MSE     : {mse_quant:.4f}")

    print("\n✅ Quantization process complete!\n")
    print("✅ Loading utils from:", utils.__file__)
    
    # Print parameter comparison table
    print("\n📌 Parameter Comparison Table")
    print(f"{'Index':<5} {'Original Coef':>18} {'Dequantized Coef':>22}")
    print("-" * 50)
    for i, (orig_c, quant_c) in enumerate(zip(coef, d_coef)):
        print(f"{i:<5} {orig_c:>18.6f} {quant_c:>22.6f}")

    print(f"\n{'Intercept':<5} {intercept:>18.6f} {d_intercept:>22.6f}")
    
    # Print prediction comparison
    print("\n📊 Prediction Comparison (first 5 samples)")
    print(f"{'Index':<5} {'Original Pred':>15} {'Quantized Pred':>20}")
    print("-" * 45)
    for i in range(5):
        print(f"{i:<5} {preds_orig[i]:>15.4f} {preds_quant[i]:>20.4f}")


if __name__ == "__main__":
    quantize_main()
