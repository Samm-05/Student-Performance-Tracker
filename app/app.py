import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st


# ABSOLUTE PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "model", "model_comparison.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ai_engine import categorize_score, generate_suggestions, risk_detection


st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_student_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_PATH):
        return None

    with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# LOAD MODEL
model = load_model()


def render_predict_page():
    st.title("Student Performance Predictor")
    st.info("Enter student learning indicators to estimate the final score.")

    input_col, insight_col = st.columns([1, 1])

    with input_col:
        st.subheader("Student Inputs")

        # INPUTS
        study_time = st.number_input("Study Time", min_value=0)
        failures = st.number_input("Past Failures", min_value=0)
        absences = st.number_input("Absences", min_value=0)

        predict_clicked = st.button("Predict Score", type="primary")

    with insight_col:
        st.subheader("Prediction Result")

        # BUTTON
        if predict_clicked:
            input_data = pd.DataFrame(
                [[study_time, failures, absences]],
                columns=["studytime", "failures", "absences"],
            )

            prediction = model.predict(input_data)
            predicted_score = float(prediction[0])
            category = categorize_score(predicted_score)
            risk_level = risk_detection(predicted_score, failures, absences)
            suggestions = generate_suggestions(
                predicted_score, study_time, failures, absences
            )

            st.success(f"Predicted Final Score: {predicted_score:.2f}")

            metric_col, risk_col = st.columns(2)
            metric_col.metric("Performance Category", category)
            risk_col.metric("Risk Level", risk_level)

            st.subheader("AI Suggestions")
            for suggestion in suggestions:
                st.info(suggestion)
        else:
            st.info("Run a prediction to view AI insights.")


def render_analytics_page():
    st.title("Student Analytics")
    st.info("Explore simple patterns from the available student dataset.")

    df = load_student_data()

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Students", len(df))
    metric_col_2.metric("Average Score", f"{df['G3'].mean():.2f}")
    metric_col_3.metric("Average Absences", f"{df['absences'].mean():.2f}")

    st.subheader("Study Time vs Score")
    st.scatter_chart(df, x="studytime", y="G3")

    st.subheader("Failures Impact Visualization")
    failures_impact = (
        df.groupby("failures", as_index=False)["G3"]
        .mean()
        .rename(columns={"G3": "average_score"})
    )
    st.bar_chart(failures_impact, x="failures", y="average_score")


def render_model_info_page():
    st.title("Model Info")
    st.info("Review training results and the selected best model.")

    comparison = load_model_comparison()

    if comparison is None:
        st.warning("Model comparison results are not available yet. Run python train.py.")
        return

    best_model = comparison.get("best_model", "Not available")
    scores = comparison.get("scores", {})
    metric = comparison.get("metric", "r2_score")

    st.success(f"Best Model: {best_model}")
    st.caption(f"Comparison metric: {metric}")

    if scores:
        scores_df = pd.DataFrame(
            [{"Model": model_name, "R2 Score": score} for model_name, score in scores.items()]
        ).sort_values("R2 Score", ascending=False)

        st.subheader("Model Comparison")
        st.dataframe(scores_df, use_container_width=True, hide_index=True)
        st.bar_chart(scores_df, x="Model", y="R2 Score")
    else:
        st.warning("No model scores were found in the comparison file.")


st.sidebar.title("Dashboard")
page = st.sidebar.radio("Navigation", ["Predict", "Analytics", "Model Info"])

if page == "Predict":
    render_predict_page()
elif page == "Analytics":
    render_analytics_page()
else:
    render_model_info_page()
