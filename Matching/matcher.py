from transformers import AutoTokenizer, RobertaModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class KeywordMatcher:
    def __init__(self, model_name="vinai/phobert-base"):
        print("Loading PhoBERT model...")
        # Remove force_download=True to use cached models when available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Fixed syntax error: force_download was missing '=True'
        # Added memory optimization parameters
        self.model = RobertaModel.from_pretrained(
            model_name,
            device_map="auto",        # Enables offloading to disk/CPU if needed
            torch_dtype=torch.float16  # Use half-precision to reduce memory usage
        )
        print("PhoBERT model loaded successfully.")
        self.model.eval()  # Set to evaluation mode

    def encode_texts(self, texts, batch_size=8):
        print("Encoding texts with PhoBERT...")
        all_embeddings = []
        
        # Process texts in smaller batches to reduce memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to free GPU memory
        
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def match_keywords_to_titles(self, keywords_df, titles_df, top_k=1, threshold=0.6):
        keyword_texts = keywords_df["normalized_keywords"].tolist()
        title_texts = (titles_df['title'] + ' ' + titles_df['title_en']).tolist()

        print(f"Processing {len(keyword_texts)} keywords and {len(title_texts)} titles...")
        
        keyword_embeddings = self.encode_texts(keyword_texts)
        title_embeddings = self.encode_texts(title_texts)

        results = []

        # Process similarities in batches to reduce memory usage
        batch_size = 100
        for i in range(0, len(keyword_embeddings), batch_size):
            batch_keywords = keyword_embeddings[i:i+batch_size]
            batch_similarities = cosine_similarity(
                batch_keywords.numpy(),
                title_embeddings.numpy()
            )
            
            for batch_idx, similarities in enumerate(batch_similarities):
                idx = i + batch_idx
                top_indices = similarities.argsort()[::-1][:top_k]

                for top_i in top_indices:
                    score = similarities[top_i]
                    if score >= threshold:
                        matched_row = titles_df.iloc[top_i]
                        results.append({
                            "keyword": keywords_df.iloc[idx]["normalized_keywords"],
                            "matched_title": matched_row.get("normalized_title", 
                                matched_row["title"] + " " + matched_row["title_en"]),
                            "original_title": matched_row["title"],
                            "original_title_en": matched_row["title_en"],
                            "category": matched_row["category"],
                            "sub_category": matched_row["sub_category"],
                            "similarity": float(score)  # Convert numpy float to Python float
                        })

        return pd.DataFrame(results)