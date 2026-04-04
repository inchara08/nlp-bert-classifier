# Transformer-Based NLP Classifier — BERT Financial Sentiment

End-to-end BERT pipeline for financial text classification on the FinancialPhraseBank dataset. Fine-tunes `bert-base-uncased` to classify financial news sentences as **positive**, **neutral**, or **negative**, directly applicable to news sentiment, earnings analysis, and market commentary.

## Results

| Model | Weighted F1 | Macro F1 | Accuracy | Notes |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.74 | 0.71 | — | — |
| BERT fine-tuned (`bert-base-uncased`) | **0.9588** | **0.9381** | **96%** | 3 epochs, lr=2e-5, held-out test set |

Per-class results on held-out test set (n=340):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.93 | 0.89 | 0.91 |
| neutral | 0.99 | 0.99 | 0.99 |
| positive | 0.91 | 0.93 | 0.92 |

## Dataset

[FinancialPhraseBank](https://huggingface.co/datasets/financial_phrasebank) — 4,845 annotated financial news sentences (positive / neutral / negative). Loaded via Hugging Face `datasets`.

## Project Structure

```
nlp-bert-classifier/
├── main.py                        # CLI entrypoint
├── config.yaml                    # hyperparameters
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── loader.py              # dataset loading + train/val/test split
│   │   └── preprocessing.py      # TextPreprocessor (financial token normalization)
│   ├── tokenization/
│   │   └── tokenizer.py          # BertTokenizer + FinancialDataset
│   ├── models/
│   │   └── classifier.py         # BertForSequenceClassification wrapper
│   ├── training/
│   │   └── trainer.py            # AdamW + warmup scheduler + early stopping
│   ├── evaluation/
│   │   └── metrics.py            # F1, precision, recall, confusion matrix
│   └── validation/
│       └── cross_validation.py   # 5-fold stratified CV
└── scripts/
    └── predict.py                 # inference on new text
```

## Quickstart

```bash
pip install -r requirements.txt

# Fine-tune BERT
python main.py --mode train

# Evaluate on held-out test set
python main.py --mode eval

# 5-fold cross-validation
python main.py --mode cv

# Predict sentiment on new text
python main.py --mode predict --text "Operating profit rose 12 percent driven by strong export volumes"
```

## Key Technical Details

- **Custom preprocessing**: HTML stripping, non-ASCII removal, financial token normalization (`Q3` → `third quarter`, `$1.2B` → `1.2 billion`, `5%` → `5 percent`)
- **Tokenization**: `BertTokenizer` with `max_length=128`, `padding="max_length"`, `truncation=True`
- **Training**: AdamW optimizer (`lr=2e-5`), linear warmup (10% steps), gradient clipping (`max_norm=1.0`), early stopping (patience=2)
- **Validation**: Stratified 5-fold CV preserving class distribution across folds
- **Model**: `bert-base-uncased` (110M parameters), fine-tuned end-to-end

## Configuration

All hyperparameters live in `config.yaml`:

```yaml
model:
  name: bert-base-uncased
  num_labels: 3
  max_length: 128

training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_clip: 1.0
  early_stopping_patience: 2

validation:
  k_folds: 5
```
