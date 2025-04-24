import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

class TitleMatcher:
    def __init__(self, model_name="vinai/phobert-base"):
        print("Loading PhoBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def preprocess_titles(self, df):
        # Combine Vietnamese + English titles
        df['title'] = df['title'].fillna('')  # Handle NaN values in 'title'
        df['title_en'] = df['title_en'].fillna('')  # Handle NaN values in 'title_en'
        df['combined_title'] = (df['title'].astype(str) + ' ' + df['title_en'].astype(str)).str.lower()
        df['combined_title'] = df['combined_title'].str.replace(
            r'[^\w\sàáạãảâấầậẫẩăắằặẵẳèéẹẽẻêếềệễểìíịĩỉòóọõỏôốồộỗổơớờợỡởùúụũủưứừựữửỳýỵỹỷđ]', '', regex=True
        )
        return df

    def encode_texts(self, texts):
        print("Encoding texts with PhoBERT...")
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings

    def match_keywords_to_titles(self, keywords, keyword_embeddings, title_df, title_embeddings, top_k=3, threshold=0.6):
        print("Matching keywords to titles...")
        matches = []
        keyword_embeddings = keyword_embeddings.cpu().numpy()
        title_embeddings = title_embeddings.cpu().numpy()

        for idx, kw_vec in enumerate(keyword_embeddings):
            sims = cosine_similarity([kw_vec], title_embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:top_k]

            for i in top_indices:
                if sims[i] >= threshold:
                    matches.append({
                        'keyword': keywords[idx],
                        'matched_title': title_df.iloc[i]['combined_title'],
                        'category': title_df.iloc[i].get('category', 'N/A'),  # Handle missing 'category' column
                        'sub_category': title_df.iloc[i].get('sub_category', 'N/A'),  # Handle missing 'sub_category' column
                        'similarity': round(float(sims[i]), 3)
                    })
        return pd.DataFrame(matches)

if __name__ == "__main__":
    # Example usage:
    matcher = TitleMatcher()

    # Load titles and keywords
    try:
        df_titles = pd.read_excel("C:/Users/ASUS/Downloads/Data/Raw/DatasetFPT.xlsx", sheet_name=7)  # Sheet with program titles
        df_keywords = pd.read_excel("C:/Users/ASUS/Downloads/Data/Raw/DatasetFPT.xlsx", sheet_name=6)  # Sheet with keywords
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit()

    # Preprocess titles
    try:
        df_titles = matcher.preprocess_titles(df_titles)
        title_embeddings = matcher.encode_texts(df_titles['combined_title'].tolist())
    except Exception as e:
        print(f"Error preprocessing or encoding titles: {e}")
        exit()

    # Process keywords
    try:
        if 'merged_keywords' not in df_keywords.columns:
            raise ValueError("The 'merged_keywords' column is missing in the keywords sheet.")
        keywords = df_keywords['merged_keywords'].dropna().unique().tolist()
        keyword_embeddings = matcher.encode_texts(keywords)
    except Exception as e:
        print(f"Error processing keywords: {e}")
        exit()

    # Match keywords to titles
    try:
        matched_df = matcher.match_keywords_to_titles(keywords, keyword_embeddings, df_titles, title_embeddings)
        matched_df.to_excel("c:/Users/ASUS/Downloads/Matched_Results.xlsx", index=False)
        print("Matching results saved.")
    except Exception as e:
        print(f"Error during matching: {e}")
