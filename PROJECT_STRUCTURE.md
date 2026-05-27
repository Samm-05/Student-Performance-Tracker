# Student Performance Tracker

AI-powered student performance analytics platform with prediction, risk
detection, model comparison, and dashboard analytics.

## Project Structure

```text
Student-Performance-Tracker/
├── app/
│   └── app.py
├── data/
│   └── student_data.csv
├── model/
│   ├── model.pkl
│   ├── best_model.pkl
│   ├── early_warning_model.pkl
│   ├── risk_model.pkl
│   ├── model_comparison.json
│   ├── model_metadata.json
│   └── shap_explanation.json
├── notebooks/
│   └── eda.ipynb
├── ai_engine.py
├── analytics.py
├── train.py
├── requirements.txt
├── requirements-optional.txt
└── README.md
```

## Run Training

```bash
python train.py
```

## Run Dashboard

```bash
streamlit run app/app.py
```
