from transformers import AutoTokenizer, RobertaModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class KeywordMatcher:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        print("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = RobertaModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()
        print("BERT model loaded successfully.")

    def encode_texts(self, texts, batch_size=8):
        print("Encoding texts with BERT...")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(batch_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def match_keywords_to_titles(self, keywords_df, titles_df, top_k=1, threshold=0.6):
        results = []

        # Normalize title sub-categories
        titles_df["sub_category"] = titles_df["sub_category"].astype(str).str.strip().str.lower()

        # Check predicted_sub_category exists
        if "predicted_sub_category" not in keywords_df.columns:
            raise ValueError("keywords_df must contain 'predicted_sub_category' column.")

        # Encode all keywords
        keyword_texts = keywords_df["normalized_keywords"].tolist()
        keyword_embeddings = self.encode_texts(keyword_texts)

        for idx, row in keywords_df.iterrows():
            keyword_text = row["normalized_keywords"]
            keyword_vec = keyword_embeddings[idx].unsqueeze(0)

            sub_cat = str(row["predicted_sub_category"]).strip().lower()
            sub_titles_df = titles_df[titles_df["sub_category"] == sub_cat]

            if sub_titles_df.empty:
                continue  # Skip unmatched sub-category

            title_texts = (sub_titles_df["title"] + " " + sub_titles_df["title_en"]).tolist()
            title_embeddings = self.encode_texts(title_texts)

            similarities = cosine_similarity(keyword_vec.numpy(), title_embeddings.numpy()).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            for top_i in top_indices:
                score = similarities[top_i]
                if score >= threshold:
                    matched_row = sub_titles_df.iloc[top_i]
                    results.append({
                        "keyword": keyword_text,
                        "matched_title": matched_row.get("normalized_title",
                                          matched_row["title"] + " " + matched_row["title_en"]),
                        "original_title": matched_row["title"],
                        "original_title_en": matched_row["title_en"],
                        "category": matched_row["category"],
                        "sub_category": matched_row["sub_category"],
                        "similarity": float(score)
                    })

        return pd.DataFrame(results)
