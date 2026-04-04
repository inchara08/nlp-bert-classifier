import pandas as pd
import torch
from src.tokenization.tokenizer import FinancialDataset, get_tokenizer


def test_dataset_length():
    df = pd.DataFrame({"text": ["hello world", "foo bar"], "label": [0, 1]})
    tokenizer = get_tokenizer()
    ds = FinancialDataset(df, tokenizer, max_length=32)
    assert len(ds) == 2


def test_dataset_keys():
    df = pd.DataFrame({"text": ["revenue increased"], "label": [2]})
    tokenizer = get_tokenizer()
    ds = FinancialDataset(df, tokenizer, max_length=32)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}


def test_tensor_shapes():
    df = pd.DataFrame({"text": ["growth exceeded expectations"], "label": [2]})
    tokenizer = get_tokenizer()
    ds = FinancialDataset(df, tokenizer, max_length=32)
    item = ds[0]
    assert item["input_ids"].shape == torch.Size([32])
    assert item["attention_mask"].shape == torch.Size([32])
    assert item["labels"].dtype == torch.long


def test_label_value():
    df = pd.DataFrame({"text": ["operating loss widened"], "label": [0]})
    tokenizer = get_tokenizer()
    ds = FinancialDataset(df, tokenizer, max_length=32)
    assert ds[0]["labels"].item() == 0


def test_attention_mask_binary():
    df = pd.DataFrame({"text": ["short text"], "label": [1]})
    tokenizer = get_tokenizer()
    ds = FinancialDataset(df, tokenizer, max_length=64)
    mask = ds[0]["attention_mask"]
    assert set(mask.unique().tolist()).issubset({0, 1})
