'''
# Civil的数据长这样：
{
    "text": "You are an idiot.", # 文本内容
    "toxicity": 0.92,            # 总体判断，这个text是不是有害言论
    "insult": 0.85,              # 侮辱性评分
    "identity_attack": 0.02,     # 人身攻击评分
    "threat": 0.01,              # 恐吓评分
    "obscene": 0.04,             # 猥亵评分
}
'''

import math

from datasets import load_dataset
import random
import torch
from torch.utils.data import Dataset

from learn_post_train.model.tokenizer import build_input

def get_train_dataset():
    dataset = load_dataset("google/civil_comments")
    return dataset['train']

def get_test_dataset():
    dataset = load_dataset("google/civil_comments")
    return dataset['test']




QUESTIONS_MAP = {
    'toxicity': "Is this a toxic message?",
    'insult': "Is this an insult?",
    'identity_attack': "Is this an identity attack?",
    'threat': "Is this a threat?",
    'obscene': "Is this obscene?",
}
LABEL_NAMES = list(QUESTIONS_MAP.keys())


class CivilDecisionDataset(Dataset):
    def __init__(self, type='train'):
        if type == 'train':
            self.dataset = get_train_dataset()
        else:
            self.dataset = get_test_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        items = []
        for label_name, question in QUESTIONS_MAP.items():
            value = example[label_name]
            if value is None or not math.isfinite(float(value)):
                # 如果某个标签数据有问题，就跳过
                continue
            # 标签、问题、标签评分
            items.append((label_name, question, float(value)))

        if not items:
            print(f"[{index}] 取数有问题， 用默认值代替")
            return ("toxicity", QUESTIONS_MAP["toxicity"], 0.0)
        
        random.shuffle(items)

        questions = [item[1] for item in items]
        targets = [item[2] for item in items]
        encoded, positions = build_input(
            example["text"],
            questions,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "decision_positions": positions,
            "labels": torch.tensor(
                targets,
                dtype=torch.float32,
            ),
        }
            
            


if __name__ == "__main__":
    dataset = CivilDecisionDataset()
    print(dataset[0])