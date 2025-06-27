from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

def predict_lables(df_keywords, tokenizer, model, label_map):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    df = df.copy()
    df["normalized_keywords"] = df["normalized_keywords"].astype(str).fillna("").str.strip()

    batch_size = 64
    texts = df["normalized_keywords"].tolist()
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            predictions.extend(batch_preds)


    df["predicted_label"]= predictions
    df["pre_sub_category"] = [label_map.get(pred, "unknown") for pred in predictions]

    print("Prediction completed")
    return df
