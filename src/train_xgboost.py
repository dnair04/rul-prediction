from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib
import os

# Paths
DATA_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load preprocessed data (fixed file name)
train_path = os.path.join(DATA_DIR, "train_FD001_clean.csv")
train = pd.read_csv(train_path)

# Features and target
X = train.drop(columns=["RUL"])
y = train["RUL"]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_val)
shap.summary_plot(shap_values, X_val)

