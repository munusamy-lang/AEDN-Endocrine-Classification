from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(
        y_true,
        y_pred,
        y_prob):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_true,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='weighted'
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class='ovr'
    )

    return {
        "Accuracy":acc,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "AUROC":auc
    }
