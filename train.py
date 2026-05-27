import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
MODEL_COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.json")
FEATURE_COLUMNS = ["studytime", "failures", "absences"]
TARGET_COLUMN = "G3"


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=100,
        )
    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost model.")

    return models


def train_and_compare_models(models, X_train, X_test, y_train, y_test):
    trained_models = {}
    model_scores = {}

    for model_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)
        model_scores[model_name] = float(r2_score(y_test, predictions))
        trained_models[model_name] = model_instance

    return trained_models, model_scores


def save_model_comparison(scores, best_model_name):
    comparison = {
        "metric": "r2_score",
        "best_model": best_model_name,
        "scores": scores,
    }

    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)


# CREATE MODEL FOLDER
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD DATA
df = pd.read_csv(DATA_PATH)

# SELECT ONLY IMPORTANT FEATURES
X = df[FEATURE_COLUMNS]

# TARGET
y = df[TARGET_COLUMN]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CREATE MODEL
model = RandomForestRegressor()

# TRAIN MODEL
model.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(model, MODEL_PATH)

# TRAIN AND COMPARE ADDITIONAL MODELS
models = get_models()
trained_models, model_scores = train_and_compare_models(
    models, X_train, X_test, y_train, y_test
)

# SELECT BEST MODEL USING R2 SCORE
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]

# SAVE BEST MODEL AND MODEL COMPARISON
joblib.dump(best_model, BEST_MODEL_PATH)
save_model_comparison(model_scores, best_model_name)

print("Model comparison:")
for model_name, score in model_scores.items():
    print(f"- {model_name}: R2 = {score:.4f}")

print(f"Best model: {best_model_name}")
print("Model Trained Successfully")
