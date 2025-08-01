import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import my_utils as utils

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data using utils
X, y = utils.load_data()

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2, mse = utils.compute_scores(y_test, y_pred)

print(f"✅ R² Score: {r2:.4f}")
print(f"✅ MSE Loss: {mse:.4f}")

# Save trained model
joblib.dump(model, "model.pth")
