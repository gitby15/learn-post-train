import torch
from torch.utils.data import Dataset

from learn_post_train.model.tokenizer import build_input


QUESTIONS = [
    "Does this comment contain an insult?",
    "Does this comment contain a threat?",
    "Is this comment toxic?",
]

SAMPLES = [
    ("You are an idiot.", [1.0, 0.0, 1.0]),
    ("I will hurt you.", [0.0, 1.0, 1.0]),
    ("Have a wonderful day!", [0.0, 0.0, 0.0]),
    ("You are stupid and worthless.", [1.0, 0.0, 1.0]),
]

# Todo: 这个小样本试验之后，是需要做一个消融实验，来验证模型是不是学到真东西了


class TinyDecisionDataset(Dataset):
    def __len__(self):
        return len(SAMPLES)

    def __getitem__(self, index):
        state, labels = SAMPLES[index]

        encoded, positions = build_input(
            state,
            QUESTIONS,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "decision_positions": positions,
            "labels": torch.tensor(
                labels,
                dtype=torch.float32,
            ),
        }