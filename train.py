from src.data import prepare_data
from src.preprocessing import preprocess_features
from src.models import create_xgb, create_logreg, create_nn, train_nn
from src.evaluation import compute_metrics, find_best_threshold
from src.explainability import create_llm_input
import torch
import numpy as np
import pandas as pd
import joblib


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        "data/heloc_dataset_v1.csv"
    )

    X_train, X_val, X_test, imputer, feature_names = preprocess_features(
        X_train,
        X_val,
        X_test
    )

    joblib.dump(
        list(feature_names),
        "artifacts/feature_names.pkl"
    )

    joblib.dump(
        imputer.statistics_,
        "artifacts/imputer_values.pkl"
    )

    print("\n===== XGBoost =====")

    xgb = create_xgb()

    xgb.fit(
        X_train,
        y_train
    )

    xgb_val_proba = xgb.predict_proba(
        X_val
    )[:, 1]

    xgb_threshold = find_best_threshold(
        y_val,
        xgb_val_proba
    )

    xgb_test_proba = xgb.predict_proba(
        X_test
    )[:, 1]

    xgb_metrics = compute_metrics(
        xgb_test_proba,
        y_test,
        threshold=xgb_threshold,
        model_name="XGBoost"
    )

    xgb.save_model(
        "artifacts/xgb_model.json"
    )

    joblib.dump(
        xgb_threshold,
        "artifacts/xgb_threshold.pkl"
    )

    print("\n===== Logistic Regression =====")

    logreg = create_logreg()

    logreg.fit(
        X_train,
        y_train
    )

    logreg_val_proba = logreg.predict_proba(
        X_val
    )[:, 1]

    logreg_threshold = find_best_threshold(
        y_val,
        logreg_val_proba
    )

    logreg_test_proba = logreg.predict_proba(
        X_test
    )[:, 1]

    logreg_metrics = compute_metrics(
        logreg_test_proba,
        y_test,
        threshold=logreg_threshold,
        model_name="Logistic Regression"
    )

    print("\n===== Neural Network =====")

    nn = create_nn(
        X_train.shape[1]
    )

    nn = train_nn(
        nn,
        X_train,
        X_val,
        y_train,
        y_val
    )

    nn.eval()

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    with torch.no_grad():
        nn_val_proba = (
            nn(X_val_tensor)
            .numpy()
            .flatten()
        )

        nn_test_proba = (
            nn(X_test_tensor)
            .numpy()
            .flatten()
        )

    nn_threshold = find_best_threshold(
        y_val,
        nn_val_proba
    )

    nn_metrics = compute_metrics(
        nn_test_proba,
        y_test,
        threshold=nn_threshold,
        model_name="Neural Network"
    )

    torch.save(
        nn.state_dict(),
        "artifacts/nn_model.pth"
    )

    joblib.dump(
        nn_threshold,
        "artifacts/nn_threshold.pkl"
    )

    results = pd.DataFrame({
        "XGBoost": xgb_metrics,
        "Logistic Regression": logreg_metrics,
        "Neural Network": nn_metrics
    }).T

    print("\n===== MODEL COMPARISON =====")
    print(results)

    results.to_csv(
        "artifacts/results.csv"
    )

    print("\nTraining finished.")

    llm_input = create_llm_input(
        xgb,
        X_test,
        feature_names,
        idx=100,
        threshold=xgb_threshold,
        top_k=5
    )

    print(llm_input)

if __name__ == '__main__':
    main()