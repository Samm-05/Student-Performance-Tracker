import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st


# ABSOLUTE PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
EARLY_WARNING_MODEL_PATH = os.path.join(BASE_DIR, "model", "early_warning_model.pkl")
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "model", "model_comparison.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ai_engine import categorize_score, generate_suggestions, risk_detection


QUICK_FEATURE_COLUMNS = ["studytime", "failures", "absences"]
ADVANCED_FEATURE_COLUMNS = [
    "G1",
    "G2",
    "age",
    "health",
    "famrel",
    "studytime",
    "failures",
    "absences",
    "attendance_rate",
    "Medu",
    "Fedu",
    "internet",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
]
EARLY_WARNING_FEATURE_COLUMNS = [
    feature for feature in ADVANCED_FEATURE_COLUMNS if feature not in ["G1", "G2"]
]


st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_optional_model(model_path):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_student_data(file_modified_time):
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_PATH):
        return None

    with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# LOAD MODEL
model = load_model()
advanced_model = load_optional_model(BEST_MODEL_PATH)
early_warning_model = load_optional_model(EARLY_WARNING_MODEL_PATH)


def get_student_data():
    return load_student_data(os.path.getmtime(DATA_PATH))


def add_prediction_insights(df):
    insights_df = df.copy()
    if advanced_model is not None and set(ADVANCED_FEATURE_COLUMNS).issubset(df.columns):
        prediction_input = insights_df[ADVANCED_FEATURE_COLUMNS]
        insights_df["predicted_score"] = advanced_model.predict(prediction_input)
    else:
        prediction_input = insights_df[QUICK_FEATURE_COLUMNS]
        insights_df["predicted_score"] = model.predict(prediction_input)
    insights_df["category"] = insights_df["predicted_score"].apply(categorize_score)
    insights_df["risk_level"] = insights_df.apply(
        lambda row: risk_detection(
            row["predicted_score"], row["failures"], row["absences"]
        ),
        axis=1,
    )
    return insights_df


def yes_no_input(label, value):
    default_index = 0 if str(value).lower() == "yes" else 1
    return st.selectbox(label, ["yes", "no"], index=default_index)


def build_input_data(values, feature_columns):
    return pd.DataFrame([[values[feature] for feature in feature_columns]], columns=feature_columns)


def render_prediction_result(
    selected_student,
    active_model,
    input_values,
    feature_columns,
    study_time,
    failures,
    absences,
    model_label,
):
    input_data = build_input_data(input_values, feature_columns)
    prediction = active_model.predict(input_data)
    predicted_score = float(prediction[0])
    category = categorize_score(predicted_score)
    risk_level = risk_detection(predicted_score, failures, absences)
    suggestions = generate_suggestions(predicted_score, study_time, failures, absences)

    st.caption(f"Student: {selected_student}")
    st.caption(f"Prediction mode: {model_label}")
    st.success(f"Predicted Final Score: {predicted_score:.2f}")

    metric_col, risk_col = st.columns(2)
    metric_col.metric("Performance Category", category)
    risk_col.metric("Risk Level", risk_level)

    st.subheader("AI Suggestions")
    for suggestion in suggestions:
        st.info(suggestion)


def render_predict_page():
    st.title("Student Performance Predictor")
    st.info("Enter student learning indicators to estimate the final score.")

    df = get_student_data()
    has_student_names = "name" in df.columns

    input_col, insight_col = st.columns([1, 1])

    with input_col:
        st.subheader("Student Inputs")

        selected_student = None
        selected_row = None

        if has_student_names:
            selected_student = st.selectbox("Student Name", df["name"].tolist())
            selected_row = df[df["name"] == selected_student].iloc[0]
        else:
            selected_student = st.text_input("Student Name", value="Student")

        prediction_mode = st.radio(
            "Prediction Mode",
            ["Advanced", "Early Warning", "Quick"],
            horizontal=True,
        )

        row = selected_row if selected_row is not None else {}

        if prediction_mode == "Quick":
            st.caption("Quick mode keeps the original 3-factor prediction flow.")
            study_time = st.number_input(
                "Study Time",
                min_value=0,
                value=int(row.get("studytime", 0)),
            )
            failures = st.number_input(
                "Past Failures",
                min_value=0,
                value=int(row.get("failures", 0)),
            )
            absences = st.number_input(
                "Absences",
                min_value=0,
                value=int(row.get("absences", 0)),
            )
            input_values = {
                "studytime": study_time,
                "failures": failures,
                "absences": absences,
            }
            active_model = model
            feature_columns = QUICK_FEATURE_COLUMNS
            model_label = "Quick Prediction"

        else:
            uses_marks = prediction_mode == "Advanced"
            if uses_marks and advanced_model is None:
                st.warning("Advanced model is not available. Run python train.py first.")
            if not uses_marks and early_warning_model is None:
                st.warning("Early-warning model is not available. Run python train.py first.")

            if uses_marks:
                st.caption("Advanced mode uses marks, attendance, activity, support, health, and family factors.")
                g1 = st.number_input("G1 Marks", min_value=0, max_value=20, value=int(row.get("G1", 0)))
                g2 = st.number_input("G2 Marks", min_value=0, max_value=20, value=int(row.get("G2", 0)))
            else:
                st.caption("Early-warning mode avoids G1/G2 and focuses on behavior and support signals.")
                g1 = None
                g2 = None

            basic_col, context_col = st.columns(2)
            with basic_col:
                age = st.number_input("Age", min_value=10, max_value=25, value=int(row.get("age", 15)))
                study_time = st.number_input("Study Time", min_value=0, value=int(row.get("studytime", 0)))
                failures = st.number_input("Past Failures", min_value=0, value=int(row.get("failures", 0)))
                absences = st.number_input("Absences", min_value=0, value=int(row.get("absences", 0)))
                attendance_rate = st.number_input(
                    "Attendance Rate (%)",
                    min_value=0,
                    max_value=100,
                    value=int(row.get("attendance_rate", max(0, 100 - int(row.get("absences", 0))))),
                )

            with context_col:
                health = st.number_input("Health", min_value=1, max_value=5, value=int(row.get("health", 3)))
                famrel = st.number_input("Family Relationship", min_value=1, max_value=5, value=int(row.get("famrel", 3)))
                medu = st.number_input("Mother Education", min_value=0, max_value=4, value=int(row.get("Medu", 0)))
                fedu = st.number_input("Father Education", min_value=0, max_value=4, value=int(row.get("Fedu", 0)))

            support_col, activity_col = st.columns(2)
            with support_col:
                internet = yes_no_input("Internet Access", row.get("internet", "yes"))
                schoolsup = yes_no_input("School Support", row.get("schoolsup", "no"))
                famsup = yes_no_input("Family Support", row.get("famsup", "no"))

            with activity_col:
                paid = yes_no_input("Paid Classes", row.get("paid", "no"))
                activities = yes_no_input("Extra Curricular Activity", row.get("activities", "no"))

            input_values = {
                "age": age,
                "health": health,
                "famrel": famrel,
                "studytime": study_time,
                "failures": failures,
                "absences": absences,
                "attendance_rate": attendance_rate,
                "Medu": medu,
                "Fedu": fedu,
                "internet": internet,
                "schoolsup": schoolsup,
                "famsup": famsup,
                "paid": paid,
                "activities": activities,
            }

            if uses_marks:
                input_values.update({"G1": g1, "G2": g2})
                active_model = advanced_model
                feature_columns = ADVANCED_FEATURE_COLUMNS
                model_label = "Advanced Prediction"
            else:
                active_model = early_warning_model
                feature_columns = EARLY_WARNING_FEATURE_COLUMNS
                model_label = "Early Warning"

        predict_clicked = st.button("Predict Score", type="primary")

    with insight_col:
        st.subheader("Prediction Result")

        # BUTTON
        if predict_clicked:
            if active_model is None:
                st.error("Selected model is not available. Run python train.py to regenerate models.")
                return

            render_prediction_result(
                selected_student,
                active_model,
                input_values,
                feature_columns,
                study_time,
                failures,
                absences,
                model_label,
            )
        else:
            st.info("Run a prediction to view AI insights.")


def render_analytics_page():
    st.title("Student Analytics")
    st.info("Explore simple patterns from the available student dataset.")

    df = get_student_data()

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

    if "name" in df.columns:
        st.subheader("Students Needing Improvement")
        prediction_df = add_prediction_insights(df)
        improvement_df = prediction_df[
            [
                "name",
                "predicted_score",
                "category",
                "risk_level",
                "G1",
                "G2",
                "attendance_rate",
                "studytime",
                "failures",
                "absences",
                "activities",
            ]
        ].sort_values(["predicted_score", "failures", "absences"], ascending=[True, False, False])

        st.warning("Students at the top of this table should be reviewed first.")
        st.dataframe(
            improvement_df.head(20),
            width="stretch",
            hide_index=True,
            column_config={
                "name": "Student Name",
                "predicted_score": st.column_config.NumberColumn(
                    "Predicted Score",
                    format="%.2f",
                ),
                "category": "Category",
                "risk_level": "Risk Level",
                "studytime": "Study Time",
                "failures": "Past Failures",
                "absences": "Absences",
                "attendance_rate": "Attendance %",
                "activities": "Extra Curricular",
            },
        )


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
        st.dataframe(scores_df, width="stretch", hide_index=True)
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
