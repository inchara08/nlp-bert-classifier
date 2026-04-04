import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from src.models.classifier import BertClassifier
from src.tokenization.tokenizer import FinancialDataset, get_tokenizer
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics


def run_cross_validation(df: pd.DataFrame, config: dict, device: torch.device) -> dict:
    k = config["validation"]["k_folds"]
    skf = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=config["validation"]["random_seed"]
    )
    tokenizer = get_tokenizer(config["model"]["name"])

    fold_metrics = []
    print(f"\nRunning {k}-fold stratified cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = FinancialDataset(train_df, tokenizer, config["model"]["max_length"])
        val_ds = FinancialDataset(val_df, tokenizer, config["model"]["max_length"])

        train_loader = DataLoader(
            train_ds, batch_size=config["training"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

        model = BertClassifier(config["model"]["name"], config["model"]["num_labels"])
        trainer = Trainer(model, config, device)
        checkpoint_dir = f"checkpoints/fold_{fold + 1}"
        trainer.train(train_loader, val_loader, checkpoint_dir=checkpoint_dir)

        model.load_state_dict(
            torch.load(
                f"{checkpoint_dir}/best_model.pt", map_location=device, weights_only=True
            )
        )
        model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        metrics = compute_metrics(all_labels, all_preds)
        fold_metrics.append(metrics)
        print(f"Fold {fold + 1} weighted F1: {metrics['weighted_f1']:.4f}")

    aggregated = aggregate_fold_metrics(fold_metrics)
    print(
        f"\nCV Results: weighted F1 = "
        f"{aggregated['weighted_f1']['mean']:.4f} ± {aggregated['weighted_f1']['std']:.4f}"
    )
    return aggregated
