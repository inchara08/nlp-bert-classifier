import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm


class Trainer:
    def __init__(self, model, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_f1 = 0.0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
    ) -> dict:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        cfg = self.config["training"]

        total_steps = len(train_loader) * cfg["epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])

        optimizer = AdamW(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        patience_counter = 0
        history = {"train_loss": [], "val_f1": []}

        for epoch in range(cfg["epochs"]):
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, cfg["gradient_clip"]
            )
            val_f1 = self._evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_f1"].append(val_f1)
            print(
                f"Epoch {epoch + 1}/{cfg['epochs']} — loss: {train_loss:.4f}  val_f1: {val_f1:.4f}"
            )

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(
                    self.model.state_dict(), f"{checkpoint_dir}/best_model.pt"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

    def _train_epoch(
        self, loader: DataLoader, optimizer, scheduler, grad_clip: float
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return f1_score(all_labels, all_preds, average="weighted")
