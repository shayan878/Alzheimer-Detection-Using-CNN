from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch.nn.functional as F


@dataclass
class EvaluationResult:
    confusion_matrix: np.ndarray
    classification_report: str
    accuracy: float
    precision: float
    auc_score: float
    jaccard: float
    recall: float
    dice: float


def get_predictions(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_scores.extend(probabilities[:, 1].cpu().numpy())

    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_scores)


def calculate_metrics(model, test_loader, device) -> EvaluationResult:
    y_true, y_pred, y_scores = get_predictions(model, test_loader, device)
    conf_mat = confusion_matrix(y_true, y_pred)

    denominator = 2 * conf_mat[1, 1] + conf_mat[1, 0] + conf_mat[0, 1]
    dice = (2 * conf_mat[1, 1] / denominator) if denominator else 0.0

    return EvaluationResult(
        confusion_matrix=conf_mat,
        classification_report=classification_report(y_true, y_pred),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        auc_score=roc_auc_score(y_true, y_pred),
        jaccard=jaccard_score(y_true, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        dice=dice,
    )


def get_roc_data(model, test_loader, device):
    y_true, _, y_scores = get_predictions(model, test_loader, device)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr, auc(fpr, tpr)
