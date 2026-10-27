import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_per_label_metrics(
    probabilities,
    soft_labels,
    label_names,
    threshold=0.5,
):
    """
    probabilities: [N, Q]
    soft_labels:   [N, Q]
    """

    results = {}

    for i, name in enumerate(label_names):
        probs = probabilities[:, i]
        soft = soft_labels[:, i]

        valid = np.isfinite(probs) & np.isfinite(soft)
        probs = probs[valid]
        soft = soft[valid]

        if len(soft) == 0:
            results[name] = {"n": 0}
            continue

        binary = (soft >= 0.5).astype(int)
        predictions = (probs >= threshold).astype(int)

        eps = 1e-7
        clipped = np.clip(probs, eps, 1 - eps)

        bce = -np.mean(
            soft * np.log(clipped)
            + (1 - soft) * np.log(1 - clipped)
        )

        # 单类别数据无法定义 AUROC
        auroc = (
            roc_auc_score(binary, probs)
            if len(np.unique(binary)) == 2
            else float("nan")
        )

        ap = (
            average_precision_score(binary, probs)
            if binary.sum() > 0
            else float("nan")
        )

        results[name] = {
            "n": len(soft),
            "positive_rate": float(binary.mean()),
            "bce": float(bce),
            "auroc": float(auroc),
            "ap": float(ap),
            "precision": float(
                precision_score(
                    binary, predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    binary, predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    binary, predictions,
                    zero_division=0,
                )
            ),
        }

    return results