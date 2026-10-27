import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from learn_post_train.dataset.civil_comment import LABEL_NAMES, CivilDecisionDataset
from learn_post_train.dataset.data_collator import MultiDecisionCollator
from learn_post_train.evaluation.metrics import (
    compute_per_label_metrics,
)
from learn_post_train.model.tokenizer import load_tokenizer


@torch.inference_mode()
def collect_predictions(model, dataset, batch_size=16):
    tokenizer = load_tokenizer()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultiDecisionCollator(tokenizer),
    )

    device = next(model.parameters()).device
    model.eval()

    all_probs = []
    all_labels = []

    for batch in tqdm(loader):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        logits = outputs["logits"]
        probs = torch.sigmoid(logits)

        # 这里假设每条样本都包含相同的 Q 个问题，
        # 且 Question 顺序固定。
        assert torch.all(batch["question_mask"] == 1)

        all_probs.append(
            probs.cpu().numpy()
        )
        all_labels.append(
            batch["labels"].cpu().numpy()
        )

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def evaluate(model):
    dataset = CivilDecisionDataset("validation")

    probs, labels = collect_predictions(
        model,
        dataset,
    )

    metrics = compute_per_label_metrics(
        probabilities=probs,
        soft_labels=labels,
        label_names=LABEL_NAMES,
    )

    for label, values in metrics.items():
        print(f"\n[{label}]")
        for key, value in values.items():
            print(f"  {key}: {value:.4f}")

    return metrics