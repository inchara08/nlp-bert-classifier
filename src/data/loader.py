import io
import yaml
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import requests

_ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank"
    "/resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
_CACHE_PATH = Path(".cache/financial_phrasebank.csv")

# Filename inside the zip for each config
_CONFIG_FILE = {
    "sentences_allagree": "Sentences_AllAgree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_50agree": "Sentences_50Agree.txt",
}

_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _fetch_data(dataset_config: str) -> pd.DataFrame:
    """Download the FinancialPhraseBank zip, parse the requested config, cache as CSV."""
    if _CACHE_PATH.exists():
        return pd.read_csv(_CACHE_PATH)

    _CACHE_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading FinancialPhraseBank from HuggingFace ...")
    r = requests.get(_ZIP_URL, timeout=60)
    r.raise_for_status()

    target_file = _CONFIG_FILE[dataset_config]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # The txt file may be nested inside a subdirectory in the zip
        matches = [n for n in zf.namelist() if n.endswith(target_file)]
        if not matches:
            raise FileNotFoundError(
                f"{target_file} not found in zip. Available: {zf.namelist()}"
            )
        raw = zf.read(matches[0]).decode("latin-1")

    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({"text": sentence.strip(), "label": label.strip()})

    df = pd.DataFrame(rows)
    df["label"] = df["label"].map(_LABEL_MAP)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(_CACHE_PATH, index=False)
    return df


def load_financial_data(config: dict) -> tuple:
    """Load FinancialPhraseBank and split into train/val/test DataFrames."""
    df = _fetch_data(config["data"]["dataset_config"])

    seed = config["validation"]["random_seed"]
    test_size = config["validation"]["test_size"]
    val_size = config["validation"]["val_size"]

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label"],
        random_state=seed,
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
