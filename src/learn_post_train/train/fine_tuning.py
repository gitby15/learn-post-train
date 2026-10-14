import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learn_post_train.dataset.tiny_test import TinyDecisionDataset
from learn_post_train.model.bert import load_model


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = load_model().to(device)
    dataset = TinyDecisionDataset()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for batch in loader:
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            positions = batch["decision_positions"].unsqueeze(0).to(device)
            labels = batch["labels"].unsqueeze(0).to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decision_positions=positions,
            )

            assert logits.shape == labels.shape

            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {total_loss / len(loader):.6f}"
            )


if __name__ == "__main__":
    train()