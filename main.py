import argparse
import torch
from torch.utils.data import DataLoader

from src.data.loader import load_config, load_financial_data
from src.data.preprocessing import TextPreprocessor
from src.tokenization.tokenizer import FinancialDataset, get_tokenizer
from src.models.classifier import BertClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics, print_report
from src.validation.cross_validation import run_cross_validation


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="BERT Financial Text Classifier")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "cv", "predict"],
        required=True,
        help="train: fine-tune BERT | eval: evaluate on test set | cv: 5-fold CV | predict: classify new text",
    )
    parser.add_argument("--text", type=str, help="Text to classify (mode=predict only)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    if args.mode == "predict":
        if not args.text:
            parser.error("--text is required for predict mode")
        from scripts.predict import predict
        result = predict(args.text, args.checkpoint, args.config)
        print(f"Prediction: {result}")
        return

    preprocessor = TextPreprocessor()
    train_df, val_df, test_df = load_financial_data(config)

    for df in [train_df, val_df, test_df]:
        df["text"] = preprocessor.batch_clean(df["text"].tolist())

    tokenizer = get_tokenizer(config["model"]["name"])

    if args.mode == "train":
        train_ds = FinancialDataset(train_df, tokenizer, config["model"]["max_length"])
        val_ds = FinancialDataset(val_df, tokenizer, config["model"]["max_length"])
        train_loader = DataLoader(
            train_ds, batch_size=config["training"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

        model = BertClassifier(config["model"]["name"], config["model"]["num_labels"])
        trainer = Trainer(model, config, device)
        trainer.train(train_loader, val_loader)
        print(f"\nTraining complete. Best val F1: {trainer.best_f1:.4f}")

    elif args.mode == "eval":
        test_ds = FinancialDataset(test_df, tokenizer, config["model"]["max_length"])
        test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

        model = BertClassifier(config["model"]["name"], config["model"]["num_labels"])
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        metrics = compute_metrics(all_labels, all_preds)
        print("\nTest Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print_report(all_labels, all_preds)

    elif args.mode == "cv":
        import pandas as pd
        full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        run_cross_validation(full_df, config, device)


if __name__ == "__main__":
    main()
