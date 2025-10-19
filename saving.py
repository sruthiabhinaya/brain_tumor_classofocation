import numpy as np
from xgboost import XGBClassifier
import joblib

# Load already extracted & selected features
X_train = np.load("X_train_selected.npy")
y_train = np.load("y_train_balanced.npy")
X_test = np.load("X_test_selected.npy")
y_test = np.load("y_test.npy")

# Train only XGBoost (fast, no CNN extraction)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Save model
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(xgb_model, "xgboost_model.joblib")
xgb_model.save_model("xgboost_model.json")
xgb_model.save_model("xgboost_model.bin")

print("XGBoost model trained & saved successfully!")

