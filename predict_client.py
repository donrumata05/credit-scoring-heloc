import pandas as pd
import joblib
import json

from dotenv import load_dotenv
import os

from src.llm import explain_prediction

from xgboost import XGBClassifier

from src.preprocessing import preprocess_features
from src.explainability import create_llm_input



def load_model():

    model = XGBClassifier()

    model.load_model(
        "artifacts/xgb_model.json"
    )

    threshold = joblib.load(
        "artifacts/xgb_threshold.pkl"
    )

    imputer = joblib.load(
        "artifacts/imputer.pkl"
    )

    feature_names = joblib.load(
        "artifacts/feature_names.pkl"
    )

    return model, threshold, imputer, feature_names


def preprocess_new_client(client, feature_names):
    special_values = [-7, -8]

    client = client.copy()

    original_features = [
        col for col in feature_names
        if not col.endswith("_special")
    ]

    client = client[original_features]

    for col in original_features:
        client[f'{col}_special'] = (
            client[col]
            .isin(special_values)
            .astype(int)
        )

    client = client.mask(
        client.isin(special_values)
    )

    client = client[feature_names]

    imputer = joblib.load(
        "artifacts/imputer.pkl"
    )

    client = imputer.transform(client)

    return pd.DataFrame(
        client,
        columns=feature_names
    )

def predict(client):
    model, threshold, imputer, feature_names = load_model()

    client_processed = preprocess_new_client(
        client,
        feature_names
    )

    client_processed = pd.DataFrame(
        imputer.transform(client_processed),
        columns=feature_names
    )

    result = create_llm_input(
        model,
        client_processed,
        feature_names,
        idx=0,
        threshold=threshold
    )

    print("===== MODEL RESULT =====")
    print(result)

    llm_explanation = explain_prediction(result)

    print("\n===== LLM EXPLANATION =====")
    print(llm_explanation)


if __name__ == "__main__":
    example_client = pd.DataFrame([{
        "ExternalRiskEstimate": 78,
        "MSinceOldestTradeOpen": 100,
        "MSinceMostRecentTradeOpen": 20,
        "AverageMInFile": 50,
        "NumSatisfactoryTrades": 30,
        "NumTrades60Ever2DerogPubRec": 0,
        "NumTrades90Ever2DerogPubRec": 0,
        "PercentTradesNeverDelq": 95,
        "MSinceMostRecentDelq": 33,
        "MaxDelq2PublicRecLast12M": 0,
        "MaxDelqEver": 0,
        "NumTotalTrades": 40,
        "NumTradesOpeninLast12M": 5,
        "PercentInstallTrades": 62,
        "MSinceMostRecentInqexcl7days": 0,
        "NumInqLast6M": 1,
        "NumInqLast6Mexcl7days": 1,
        "NetFractionRevolvingBurden": 8,
        "NetFractionInstallBurden": 100,
        "NumRevolvingTradesWBalance": 5,
        "NumInstallTradesWBalance": 3,
        "NumBank2NatlTradesWHighUtilization": 0,
        "PercentTradesWBalance": 50
    }])

    result = predict(example_client)

    print(
        json.dumps(
            result,
            indent=4
        )
    )