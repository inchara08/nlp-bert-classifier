import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from transformers import BertTokenizer

from src.models.classifier import BertClassifier
from src.data.preprocessing import TextPreprocessor
from src.data.loader import load_config

LABEL_NAMES = ["negative", "neutral", "positive"]


def predict(
    text: str,
    checkpoint: str = "checkpoints/best_model.pt",
    config_path: str = "config.yaml",
) -> str:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = TextPreprocessor()
    tokenizer = BertTokenizer.from_pretrained(config["model"]["name"])
    model = BertClassifier(config["model"]["name"], config["model"]["num_labels"])
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    cleaned = preprocessor.clean(text)
    encoding = tokenizer(
        cleaned,
        max_length=config["model"]["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )
    pred = outputs.logits.argmax(dim=-1).item()
    return LABEL_NAMES[pred]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for financial text")
    parser.add_argument("--text", required=True, help="Financial text to classify")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    result = predict(args.text, args.checkpoint, args.config)
    print(f"Prediction: {result}")
