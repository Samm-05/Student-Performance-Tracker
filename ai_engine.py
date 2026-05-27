def categorize_score(score):
    if score >= 16:
        return "Excellent"
    if score >= 12:
        return "Good"
    if score >= 8:
        return "Average"
    return "At Risk"


def generate_suggestions(score, study_time, failures, absences):
    suggestions = []

    if score < 8:
        suggestions.append("Create a weekly recovery plan focused on core weak topics.")
        suggestions.append("Schedule regular check-ins with a teacher or academic mentor.")
    elif score < 12:
        suggestions.append("Review recent mistakes and practice similar questions daily.")
    elif score < 16:
        suggestions.append("Maintain consistent revision and attempt timed practice tests.")
    else:
        suggestions.append("Keep the current study routine and focus on advanced practice.")

    if study_time < 2:
        suggestions.append("Increase study time to at least 2 focused sessions per week.")

    if failures > 0:
        suggestions.append("Prioritize subjects with past failures and track weekly progress.")

    if absences > 5:
        suggestions.append("Reduce absences and catch up on missed lessons quickly.")

    return suggestions


def risk_detection(score, failures, absences):
    if score < 8 or failures >= 2 or absences > 10:
        return "High Risk"
    if score < 12 or failures == 1 or absences > 5:
        return "Medium Risk"
    return "Low Risk"


def explain_prediction(input_data, model):
    feature_names = list(getattr(input_data, "columns", []))

    if not feature_names:
        return {
            "summary": "Feature names are unavailable for this prediction.",
            "feature_importance": {},
        }

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = abs(model.coef_)
    else:
        return {
            "summary": "This model does not expose feature importance values.",
            "feature_importance": {},
        }

    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(feature_names, importance_values)
    }
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    top_feature = next(iter(feature_importance), None)
    summary = (
        f"The prediction was most influenced by {top_feature}."
        if top_feature
        else "No dominant feature could be identified."
    )

    return {
        "summary": summary,
        "feature_importance": feature_importance,
    }
