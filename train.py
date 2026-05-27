import json
import os
from importlib import import_module
from importlib.util import find_spec
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
EARLY_WARNING_MODEL_PATH = os.path.join(MODEL_DIR, "early_warning_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
MODEL_COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.json")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
SHAP_EXPLANATION_PATH = os.path.join(MODEL_DIR, "shap_explanation.json")

FEATURE_COLUMNS = ["studytime", "failures", "absences"]
TARGET_COLUMN = "G3"
ADVANCED_NUMERIC_FEATURES = [
    "G1",
    "G2",
    "age",
    "health",
    "famrel",
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
]
ADVANCED_CATEGORICAL_FEATURES = [
    "internet",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
]
ADVANCED_FEATURE_COLUMNS = ADVANCED_NUMERIC_FEATURES + ADVANCED_CATEGORICAL_FEATURES
EARLY_WARNING_FEATURE_COLUMNS = [
    feature for feature in ADVANCED_FEATURE_COLUMNS if feature not in ["G1", "G2"]
]
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
RISK_THRESHOLD = 8


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
    }

    if find_spec("xgboost") is not None:
        XGBRegressor = import_module("xgboost").XGBRegressor

        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=100,
        )
    else:
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


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_pipeline(model, numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def get_advanced_model_searches(numeric_features, categorical_features):
    searches = {
        "Linear Regression": (
            build_pipeline(LinearRegression(), numeric_features, categorical_features),
            {},
        ),
        "Decision Tree": (
            build_pipeline(
                DecisionTreeRegressor(random_state=42),
                numeric_features,
                categorical_features,
            ),
            {
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "Random Forest": (
            build_pipeline(
                RandomForestRegressor(random_state=42),
                numeric_features,
                categorical_features,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }

    if find_spec("xgboost") is not None:
        XGBRegressor = import_module("xgboost").XGBRegressor

        searches["XGBoost"] = (
            build_pipeline(
                XGBRegressor(objective="reg:squarederror", random_state=42),
                numeric_features,
                categorical_features,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    else:
        print("XGBoost is not installed. Skipping advanced XGBoost tuning.")

    return searches


def tune_and_validate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    numeric_features,
    categorical_features,
    cv_folds=5,
):
    results = {}
    trained_models = {}

    for model_name, (pipeline, param_grid) in get_advanced_model_searches(
        numeric_features, categorical_features
    ).items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="r2",
            cv=cv_folds,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        results[model_name] = {
            "r2_score": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(rmse),
            "cross_validation_r2_mean": float(search.best_score_),
            "best_params": search.best_params_,
        }
        trained_models[model_name] = best_model

    best_model_name = max(results, key=lambda name: results[name]["r2_score"])
    return trained_models, results, best_model_name


def train_early_warning_model(df):
    early_numeric_features = [
        feature
        for feature in ADVANCED_NUMERIC_FEATURES
        if feature in EARLY_WARNING_FEATURE_COLUMNS
    ]
    early_categorical_features = [
        feature
        for feature in ADVANCED_CATEGORICAL_FEATURES
        if feature in EARLY_WARNING_FEATURE_COLUMNS
    ]
    X = df[EARLY_WARNING_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline(
        RandomForestRegressor(random_state=42, n_estimators=200),
        early_numeric_features,
        early_categorical_features,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    metrics = {
        "r2_score": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "cross_validation_r2_mean": float(cv_scores.mean()),
    }

    return model, metrics


def train_risk_probability_model(df):
    X = df[ADVANCED_FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] < RISK_THRESHOLD).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(
        RandomForestClassifier(random_state=42, n_estimators=200),
        ADVANCED_NUMERIC_FEATURES,
        ADVANCED_CATEGORICAL_FEATURES,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "risk_threshold": RISK_THRESHOLD,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    return model, metrics


def create_shap_explanation(model, X_sample):
    if find_spec("shap") is None:
        return {
            "available": False,
            "reason": "SHAP is not installed. Install shap to generate SHAP explanations.",
        }

    try:
        shap = import_module("shap")

        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)
        mean_absolute_values = abs(shap_values.values).mean(axis=0)
        importance = {
            feature: float(value)
            for feature, value in zip(X_sample.columns, mean_absolute_values)
        }

        return {
            "available": True,
            "method": "shap.Explainer",
            "mean_absolute_shap_values": dict(
                sorted(importance.items(), key=lambda item: item[1], reverse=True)
            ),
        }
    except Exception as error:
        return {
            "available": False,
            "reason": f"SHAP explanation failed: {error}",
        }


def save_model_comparison(scores, best_model_name, advanced_results=None):
    comparison = {
        "metric": "r2_score",
        "best_model": best_model_name,
        "scores": scores,
    }

    if advanced_results is not None:
        comparison["advanced_results"] = advanced_results

    save_json(MODEL_COMPARISON_PATH, comparison)


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

# TRAIN AND COMPARE ADDITIONAL LEGACY MODELS
models = get_models()
trained_models, model_scores = train_and_compare_models(
    models, X_train, X_test, y_train, y_test
)

# TRAIN ADVANCED PRODUCTION PIPELINE
X_advanced = df[ADVANCED_FEATURE_COLUMNS]
y_advanced = df[TARGET_COLUMN]
X_advanced_train, X_advanced_test, y_advanced_train, y_advanced_test = (
    train_test_split(X_advanced, y_advanced, test_size=0.2, random_state=42)
)
advanced_models, advanced_results, advanced_best_model_name = tune_and_validate_models(
    X_advanced_train,
    X_advanced_test,
    y_advanced_train,
    y_advanced_test,
    ADVANCED_NUMERIC_FEATURES,
    ADVANCED_CATEGORICAL_FEATURES,
)
advanced_best_model = advanced_models[advanced_best_model_name]

# TRAIN EARLY-WARNING MODEL WITHOUT G1/G2
early_warning_model, early_warning_metrics = train_early_warning_model(df)

# TRAIN RISK PROBABILITY MODEL
risk_model, risk_model_metrics = train_risk_probability_model(df)

# SAVE BEST MODELS
joblib.dump(advanced_best_model, BEST_MODEL_PATH)
joblib.dump(early_warning_model, EARLY_WARNING_MODEL_PATH)
joblib.dump(risk_model, RISK_MODEL_PATH)

# OPTIONAL SHAP EXPLAINABILITY ARTIFACT
shap_explanation = create_shap_explanation(
    advanced_best_model, X_advanced_test.head(50).reset_index(drop=True)
)
save_json(SHAP_EXPLANATION_PATH, shap_explanation)

# SAVE MODEL COMPARISON FOR EXISTING DASHBOARD COMPATIBILITY
advanced_score_summary = {
    model_name: metrics["r2_score"] for model_name, metrics in advanced_results.items()
}
save_model_comparison(
    advanced_score_summary,
    advanced_best_model_name,
    advanced_results=advanced_results,
)

# SAVE PRODUCTION METADATA
best_metrics = advanced_results[advanced_best_model_name]
confidence_score = max(0.0, min(1.0, best_metrics["r2_score"]))
metadata = {
    "model_version": MODEL_VERSION,
    "created_at": datetime.now().isoformat(),
    "target": TARGET_COLUMN,
    "legacy_model": {
        "path": MODEL_PATH,
        "features": FEATURE_COLUMNS,
        "purpose": "Backward-compatible Streamlit predictor.",
        "scores": model_scores,
    },
    "advanced_model": {
        "path": BEST_MODEL_PATH,
        "best_model": advanced_best_model_name,
        "features": ADVANCED_FEATURE_COLUMNS,
        "uses_prior_grades": True,
        "metrics": best_metrics,
        "confidence_score": float(confidence_score),
    },
    "early_warning_model": {
        "path": EARLY_WARNING_MODEL_PATH,
        "features": EARLY_WARNING_FEATURE_COLUMNS,
        "uses_prior_grades": False,
        "metrics": early_warning_metrics,
    },
    "risk_probability_model": {
        "path": RISK_MODEL_PATH,
        "features": ADVANCED_FEATURE_COLUMNS,
        "positive_class": f"{TARGET_COLUMN} < {RISK_THRESHOLD}",
        "metrics": risk_model_metrics,
    },
    "shap_explainability": shap_explanation,
}
save_json(MODEL_METADATA_PATH, metadata)

print("Legacy model comparison:")
for model_name, score in model_scores.items():
    print(f"- {model_name}: R2 = {score:.4f}")

print("\nAdvanced model comparison:")
for model_name, metrics in advanced_results.items():
    print(
        f"- {model_name}: "
        f"R2 = {metrics['r2_score']:.4f}, "
        f"CV R2 = {metrics['cross_validation_r2_mean']:.4f}, "
        f"MAE = {metrics['mae']:.4f}"
    )

print(f"\nBest advanced model: {advanced_best_model_name}")
print(f"Early-warning model R2: {early_warning_metrics['r2_score']:.4f}")
print(f"Risk model ROC AUC: {risk_model_metrics['roc_auc']:.4f}")
print("Model Trained Successfully")
