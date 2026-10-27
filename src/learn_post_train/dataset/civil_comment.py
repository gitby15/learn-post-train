import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from learn_post_train.dataset.common import TEST_QUESTION_MAP, TRAIN_QUESTIONS_MAP


def get_dataset(split: str = 'test', sort: bool = True):
    if split not in ['test', 'train', 'validation']:
        raise ValueError(f"split must be one of ['test', 'train', 'validation'], but got {split}")
    dataset = load_dataset("google/civil_comments")[split]

    if split == 'validation':
        dataset = dataset.take(1000)

    if sort:
        dataset = dataset.map(
            lambda batch: {"text_length": [len(text) for text in batch["text"]]},
            batched=True,
        )
        dataset = dataset.sort("text_length", reverse=True).remove_columns("text_length")
    return dataset





LABEL_NAMES = list(TRAIN_QUESTIONS_MAP.keys())


class CivilCommentDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = get_dataset(split)
        
        if split == 'test':
            self.labels = TEST_QUESTION_MAP.keys()
            self.question_map = TEST_QUESTION_MAP
        else:
            self.labels = TRAIN_QUESTIONS_MAP.keys()
            self.question_map = TRAIN_QUESTIONS_MAP
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        state = example['text']
        questions = []
        labels = []
        for label in self.labels:
            score = example[label]
            if score is None:
                continue
            available_questions = self.question_map[label]
            selected_questions = random.sample(
                available_questions,
                k=2,
            )
            questions.extend(selected_questions)
            labels.extend([score] * len(selected_questions))
        return {
            'state': state,
            'questions': questions,
            'labels': labels
        }



class CivilCommentCollector:
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
    dataset = CivilCommentDataset()
    for item in dataset:
        print(f"len: {len(item['state'])}")
