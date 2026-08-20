import joblib
import pandas as pd
from pathlib import Path

import xgboost as xgb

from src.explainability import create_llm_input
from src.llm import explain_prediction


ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


class CreditScoringInference:

    def __init__(self):

        self.model = xgb.XGBClassifier()
        self.model.load_model(
            ARTIFACTS_DIR / "xgb_model.json"
        )

        self.threshold = joblib.load(
            ARTIFACTS_DIR / "xgb_threshold.pkl"
        )

        self.imputer = joblib.load(
            ARTIFACTS_DIR / "imputer.pkl"
        )

        self.feature_names = joblib.load(
            ARTIFACTS_DIR / "feature_names.pkl"
        )


    def preprocess(self, client_df):

        special_values = [-7, -8]

        client_df = client_df.copy()


        original_features = [
            col
            for col in self.feature_names
            if not col.endswith("_special")
        ]


        client_df = client_df[original_features]


        for col in original_features:

            client_df[f"{col}_special"] = (
                client_df[col]
                .isin(special_values)
                .astype(int)
            )


        client_df = client_df.mask(
            client_df.isin(special_values)
        )


        client_df = client_df[self.feature_names]


        client_df = self.imputer.transform(
            client_df
        )


        return pd.DataFrame(
            client_df,
            columns=self.feature_names
        )


    def predict(self, client_df):

        processed_client = self.preprocess(
            client_df
        )


        result = create_llm_input(
            self.model,
            processed_client,
            self.feature_names,
            idx=0,
            threshold=self.threshold,
            top_k=5
        )


        explanation = explain_prediction(
            result
        )


        return {
            "prediction": result["prediction"],
            "probability": float(
                result["probability"]
            ),
            "threshold": float(
                result["threshold"]
            ),
            "top_factors": result["top_factors"],
            "llm_explanation": explanation
        }


credit_model = CreditScoringInference()