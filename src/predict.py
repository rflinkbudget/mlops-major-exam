import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("model.pth")

# Load data
data = fetch_california_housing()
_, X_test, _, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test[:5])
print("✅ Sample Predictions:", y_pred)
