from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.inference import credit_model

app = FastAPI(
    title="Credit Scoring API",
    description="Credit scoring with XGBoost, SHAP and GigaChat explanation",
    version="1.0.0"
)

class ClientData(BaseModel):

    ExternalRiskEstimate: float
    MSinceOldestTradeOpen: float
    MSinceMostRecentTradeOpen: float
    AverageMInFile: float
    NumSatisfactoryTrades: float
    NumTrades60Ever2DerogPubRec: float
    NumTrades90Ever2DerogPubRec: float
    PercentTradesNeverDelq: float
    MSinceMostRecentDelq: float
    MaxDelq2PublicRecLast12M: float
    MaxDelqEver: float
    NumTotalTrades: float
    NumTradesOpeninLast12M: float
    PercentInstallTrades: float
    MSinceMostRecentInqexcl7days: float
    NumInqLast6M: float
    NumInqLast6Mexcl7days: float
    NetFractionRevolvingBurden: float
    NetFractionInstallBurden: float
    NumRevolvingTradesWBalance: float
    NumInstallTradesWBalance: float
    NumBank2NatlTradesWHighUtilization: float
    PercentTradesWBalance: float

class PredictionResponse(BaseModel):

    status: str
    prediction: str
    probability: float
    threshold: float
    top_factors: list
    llm_explanation: str



@app.get("/")
def root():

    return {
        "message": "Credit Scoring API is running"
    }



@app.get("/health")
def health():

    return {
        "status": "ok"
    }



@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_endpoint(client: ClientData):

    try:

        client_df = pd.DataFrame(
            [client.model_dump()]
        )


        result = credit_model.predict(
            client_df
        )


        return {
            "status": "success",
            **result
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

