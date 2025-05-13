from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import torch
from Modeling.model_loader import ModelLoader

def predict_lables(df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()  # Set model to evaluation mode

    texta = df["normalized_keywords"].tolist()
    inputs = tokenizer(
        texta,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

    df["predicted_sub_category"] = predictions.numpy()
    return df

