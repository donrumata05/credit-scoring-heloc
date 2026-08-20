import shap
import pandas as pd


def explain_client(model, X, idx):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    shap.plots.waterfall(shap_values[idx])

    explanation = pd.DataFrame({
        'feature': X.columns,
        'value': X.iloc[idx].values,
        'shap_value': shap_values[idx].values
    })
    explanation['abs_shap'] = explanation['shap_value'].abs()

    explanation = explanation.sort_values(
        'abs_shap',
        ascending=False
    )

    return explanation


def create_llm_input(model, X, feature_names, idx, threshold, top_k=5):
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X)

    probability = float(
        model.predict_proba(X.iloc[[idx]])[0, 1]
    )

    prediction = "Bad" if probability > threshold else "Good"

    explanation = pd.DataFrame({
        'feature': feature_names,
        'value': X.iloc[idx].values,
        'shap': shap_explanation[idx].values
    })

    explanation['abs_shap'] = explanation['shap'].abs()

    explanation = explanation.sort_values(
        'abs_shap',
        ascending=False
    )

    top_factors = []

    for _, row in explanation.head(top_k).iterrows():
        top_factors.append({
            'feature': row['feature'],
            'value': float(row['value']),
            'shap': float(row['shap'])
        })

    return {
        'prediction': prediction,
        'probability': probability,
        'threshold': float(threshold),
        'top_factors': top_factors
    }