import torch.nn as nn
from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def freeze_base(self):
        for param in self.bert.bert.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.bert.bert.parameters():
            param.requires_grad = True
