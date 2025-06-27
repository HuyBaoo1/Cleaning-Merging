# Required imports
import os
import sys
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from openpyxl import load_workbook
from pandas import ExcelWriter


class ModelLoader:
    def __init__(self, model_path="distilbert-base-multilingual-cased", device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device).eval()
        return self.tokenizer, self.model

    def save_model(self, model, tokenizer):
        tokenizer.save_pretrained(self.model_path)
        model.save_pretrained(self.model_path)


def train_classifier(titles_df, model_name="distilbert-base-multilingual-cased", output_dir="./Modeling/model_output"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    label_encoder = LabelEncoder()
    titles_df["label"] = label_encoder.fit_transform(titles_df["sub_category"])

    dataset = Dataset.from_pandas(titles_df[["normalized_title", "label"]].rename(columns={"normalized_title": "text"}))
    dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256), batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        num_train_epochs=5,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1,
        save_strategy="epoch",
        logging_steps=1000,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    return output_dir, label_encoder


def predict_labels(df, tokenizer, model, label_map):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    df = df.copy()
    texts = df["normalized_keywords"].astype(str).fillna("").tolist()

    predictions = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            predictions.extend(batch_preds)

    df["predicted_label"] = predictions
    df["pre_sub_category"] = [label_map.get(pred, "unknown") for pred in predictions]
    return df


class KeywordMatcher:
    def __init__(self, tokenizer, model, title_embeddings, titles_df, device, fallback_to_all_titles=True):
        self.tokenizer = tokenizer
        self.model = model
        self.title_embeddings = title_embeddings
        self.titles_df = titles_df
        self.device = device
        self.fallback_to_all_titles = fallback_to_all_titles

    def encode_texts(self, texts, batch_size=32):
        all_embeddings = []
        self.model.to(self.device).eval()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings)

    def match_keywords_to_titles(self, keywords_df, top_k=1, threshold=0.75):
        keywords_df["pre_sub_category"] = keywords_df["pre_sub_category"].str.strip().str.lower()
        self.titles_df["sub_category"] = self.titles_df["sub_category"].str.strip().str.lower()

        keyword_embeddings = self.encode_texts(keywords_df["normalized_keywords"].tolist())
        results = []

        for idx, row in keywords_df.iterrows():
            keyword_text = row["normalized_keywords"]
            keyword_vec = keyword_embeddings[idx].unsqueeze(0)
            sub_cat = row["pre_sub_category"]

            sub_indices = self.titles_df.index[self.titles_df["sub_category"] == sub_cat].tolist()
            current_embeddings = self.title_embeddings[sub_indices] if sub_indices else self.title_embeddings
            current_titles_df = self.titles_df.iloc[sub_indices] if sub_indices else self.titles_df

            similarities = cosine_similarity(keyword_vec.numpy(), current_embeddings.numpy()).flatten()

            above_threshold_indices = [i for i, score in enumerate(similarities) if score >= threshold]
            if not above_threshold_indices and not self.fallback_to_all_titles:
                continue

            sorted_indices = np.argsort(similarities[above_threshold_indices])[::-1][:top_k]
            top_indices = [above_threshold_indices[i] for i in sorted_indices]

            for rel_idx in top_indices:
                matched_row = current_titles_df.iloc[rel_idx]
                sim_score = similarities[rel_idx]
                results.append({
                    "keyword": keyword_text,
                    "matched_title": matched_row.get("normalized_title", matched_row["title"] + " " + matched_row["title_en"]),
                    "original_title": matched_row["title"],
                    "original_title_en": matched_row["title_en"],
                    "category": matched_row["category"],
                    "sub_category": matched_row["sub_category"],
                    "pre_sub_category": sub_cat,
                    "similarity": float(sim_score)
                })

        return pd.DataFrame(results)


def load_clean_data(file_path, keyword_sheet, title_sheet):
    keywords_df = pd.read_excel(file_path, sheet_name=keyword_sheet)
    titles_df = pd.read_excel(file_path, sheet_name=title_sheet)
    return keywords_df, titles_df


def prepare_data(keywords_df, titles_df):
    keywords_df['normalized_keywords'] = keywords_df['normalized_keywords'].fillna('').astype(str).str.strip()
    titles_df['normalized_title'] = titles_df['normalized_title'].fillna('').astype(str).str.strip()
    titles_df['title'] = titles_df['title'].fillna('').astype(str).str.strip()
    titles_df['title_en'] = titles_df['title_en'].fillna('').astype(str).str.strip()
    titles_df['category'] = titles_df['category'].fillna('').astype(str)
    titles_df['sub_category'] = titles_df['sub_category'].fillna('').astype(str)
    return keywords_df, titles_df


def encode_titles(titles, tokenizer, model, device):
    model.to(device).eval()
    embeddings = []
    for i in range(0, len(titles), 32):
        batch = titles[i:i + 32]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu()
            embeddings.append(emb)
    return torch.cat(embeddings)


def aggregate_results(df):
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["category", "sub_category"]).agg(
        matched_keywords_count=("keyword", "count"),
        avg_similarity=("similarity", "mean"),
        min_similarity=("similarity", "min"),
        max_similarity=("similarity", "max")
    ).reset_index()


def save_to_excel(df, file_path, sheet_name):
    with ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def main():
    file_path = "C:/Users/ASUS/Downloads/Data/Raw/DatasetFPT.xlsx"
    model_path = "d:/FPT-Cleaning-Data/Modeling/model_output"
    keyword_sheet, title_sheet = "Clean Keyword", "Clean Title"

    df_keywords, df_titles = load_clean_data(file_path, keyword_sheet, title_sheet)
    df_keywords, df_titles = prepare_data(df_keywords, df_titles)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
    label_map = {i: l for i, l in enumerate(label_encoder.classes_)}

    df_keywords = predict_labels(df_keywords, tokenizer, model, label_map)

    base_model = getattr(model, "base_model", model)
    title_embeddings = encode_titles(df_titles["normalized_title"].tolist(), tokenizer, base_model, device)

    matcher = KeywordMatcher(tokenizer, base_model, title_embeddings, df_titles, device)
    matched_df = matcher.match_keywords_to_titles(df_keywords)

    summary_df = aggregate_results(matched_df)

    save_to_excel(matched_df, file_path, "Matched-Result")
    save_to_excel(summary_df, file_path, "Summary")

    print(f"Matching completed. Total matched: {len(matched_df)}")


if __name__ == "__main__":
    main()
