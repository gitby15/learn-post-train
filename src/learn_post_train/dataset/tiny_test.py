import torch
from torch.utils.data import Dataset

QUESTIONS = [
    "Does this comment contain an insult?",
    "Does this comment contain a threat?",
    "Is this comment obsence?",
]

SAMPLES = [
    ("You are an idiot.", [1.0, 0.0, 0.0]),
    ("I will kill you.", [0.0, 1.0, 0.0]),
    ("Have a wonderful day!", [0.0, 0.0, 0.0]),
    ("You are stupid and worthless.", [1.0, 0.0, 0.0]),
    ("Come to blow my dicks.", [0.0, 0.0, 1.0]),
]

# Todo: 这个小样本试验之后，是需要做一个消融实验，来验证模型是不是学到真东西了

class TinyDecisionDataset(Dataset):
    def __len__(self):
        return len(SAMPLES)

    def __getitem__(self, index):
        state, labels = SAMPLES[index]

        return {
            "state": state,
            "questions": QUESTIONS,
            "labels": labels
        }

class TinyCollector:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    def __call__(self, features):
        # 给出的形状，是[Q, L]
        s_list = []
        q_list = []
        l_list = []
        for f in features:
            state = f["state"]
            questions = f["questions"]
            labels = f["labels"]
            for q,l in zip(questions, labels):
                s_list.append(state)
                q_list.append(q)
                l_list.append(l)

        encoded = self.tokenizer(
            s_list,
            q_list,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': torch.tensor(l_list, dtype=torch.float32),
        }


if __name__ == "__main__":
    dataset = TinyDecisionDataset()
    print(dataset[0])
