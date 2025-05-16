import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os

class TitleMatcher:
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        print("Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def preprocess_titles(self, df):
        df['title'] = df['title'].fillna('')
        df['title_en'] = df['title_en'].fillna('')
        df['combined_title'] = (df['title'].astype(str) + ' ' + df['title_en'].astype(str)).str.lower()
        df['combined_title'] = df['combined_title'].str.replace(
            r'[^\w\sàáạãảâấầậẫẩăắằặẵẳèéẹẽẻêếềệễểìíịĩỉòóọõỏôốồộỗổơớờợỡởùúụũủưứừựữửỳýỵỹỷđ]',
            '', regex=True
        )
        return df

    def encode_texts(self, texts, batch_size=16):
        print("Encoding texts with BERT...")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def compute_subcategory_centroids(self, title_df, title_embeddings):
        print("Computing sub-category centroids...")
        title_df = title_df.copy()
        title_df['embedding'] = list(title_embeddings.cpu().numpy())
        title_df['sub_category'] = title_df['sub_category'].astype(str).str.strip().str.lower()

        centroids = (
            title_df.groupby('sub_category')['embedding']
            .apply(lambda x: torch.tensor(np.stack(x.values)).mean(dim=0))
            .to_dict()
        )
        return centroids

    def classify_keywords_by_subcategory(self, keywords, keyword_embeddings, centroids):
        print("Classifying keywords to sub-categories...")
        results = []
        subcats = list(centroids.keys())
        centroid_matrix = torch.stack([centroids[sub] for sub in subcats])
        keyword_embeddings = keyword_embeddings.cpu()

        for idx, kw_vec in enumerate(keyword_embeddings):
            sims = cosine_similarity(
                kw_vec.unsqueeze(0).numpy(),
                centroid_matrix.numpy()
            )[0]
            best_idx = np.argmax(sims)
            results.append({
                "keyword": keywords[idx],
                "predicted_sub_category": subcats[best_idx],
                "similarity": round(float(sims[best_idx]), 3)
            })

        return pd.DataFrame(results)

if __name__ == "__main__":
    matcher = TitleMatcher()

    # Define file paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(root_dir, "Data", "Raw", "DatasetFPT.xlsx")
    output_path = os.path.join(root_dir, "Output", "Keyword_Subcategory_Classification.xlsx")

    # Load titles and keywords
    try:
        df_titles = pd.read_excel(raw_path, sheet_name=7)
        df_keywords = pd.read_excel(raw_path, sheet_name=6)
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        exit()

    # Preprocess and encode titles
    try:
        df_titles = matcher.preprocess_titles(df_titles)
        title_embeddings = matcher.encode_texts(df_titles['combined_title'].tolist())
    except Exception as e:
        print(f"❌ Error preprocessing or encoding titles: {e}")
        exit()

    # Process keywords
    try:
        if 'merged_keywords' not in df_keywords.columns:
            raise ValueError("The 'merged_keywords' column is missing in the keywords sheet.")
        keywords = df_keywords['merged_keywords'].dropna().unique().tolist()
        keyword_embeddings = matcher.encode_texts(keywords)
    except Exception as e:
        print(f"❌ Error processing keywords: {e}")
        exit()

    # Classify keywords to sub-categories
    try:
        centroids = matcher.compute_subcategory_centroids(df_titles, title_embeddings)
        classified_df = matcher.classify_keywords_by_subcategory(keywords, keyword_embeddings, centroids)
        classified_df.to_excel(output_path, index=False)
        print(f"✅ Sub-category classification results saved to {output_path}")
    except Exception as e:
        print(f"❌ Error during classification: {e}")
