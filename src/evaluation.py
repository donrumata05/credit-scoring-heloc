from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np


def compute_metrics(predictions, labels, threshold=0.5, model_name=None):
    predictions_binary = (predictions > threshold).astype(int)
    accuracy = accuracy_score(labels, predictions_binary)
    f1 = f1_score(labels, predictions_binary)
    precision = precision_score(labels, predictions_binary)
    recall = recall_score(labels, predictions_binary)
    roc_auc = roc_auc_score(labels, predictions)
    pr_auc = average_precision_score(labels, predictions)

    print(f'Metrics for {model_name}:')
    print()
    print(f'Threshold: {threshold:.2f}')
    print(f'F1: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'ROC-AUC: {roc_auc:.2f}')
    print(f'PR-AUC: {pr_auc:.2f}')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc-auc': roc_auc,
        'pr-auc': pr_auc
    }


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-8)
    )
    best_idx = np.argmax(f1_scores)
    print(f'Best Threshold: {thresholds[best_idx]}')
    return thresholds[best_idx]

