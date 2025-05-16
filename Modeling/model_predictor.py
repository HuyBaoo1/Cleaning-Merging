import torch
from Modeling.model_loader import ModelLoader

def predict_lables(df, model_path):
    loader = ModelLoader(model_path)
    tokenizer, model = loader.load_model()

    texts = df["normalized_keywords"].astype(str).tolist()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

    df["predicted_sub_category"] = predictions
    return df


