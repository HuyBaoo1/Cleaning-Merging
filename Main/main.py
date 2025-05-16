import sys
import os
import pandas as pd

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Matching.matcher import KeywordMatcher
from Matching.result_aggregator import aggregate_results
from Output.saver import save_to_excel
from Modeling.model_predictor import predict_lables  

def load_clean_data(file_path, keyword_sheet, title_sheet):
    print("Loading cleaned data...")
    try:
        keywords_df = pd.read_excel(file_path, sheet_name=keyword_sheet)
        titles_df = pd.read_excel(file_path, sheet_name=title_sheet)
        return keywords_df, titles_df
    except Exception as e:
        raise RuntimeError(f"Failed to load Excel sheets: {e}")

def prepare_data(keywords_df, titles_df):

    print("Preprocessing data...")

    # Clean keyword data
    keywords_df['normalized_keywords'] = keywords_df.get('normalized_keywords', '').fillna('').astype(str).str.strip()

    # Clean title data
    titles_df['title'] = titles_df.get('title', '').fillna('').astype(str).str.strip()
    titles_df['title_en'] = titles_df.get('title_en', '').fillna('').astype(str).str.strip()
    titles_df['normalized_title'] = titles_df.get('normalized_title', '').fillna('').astype(str).str.strip()
    titles_df['combined_title'] = (titles_df['title'] + ' ' + titles_df['title_en']).str.lower()
    titles_df['sub_category'] = titles_df.get('sub_category', '').fillna('').astype(str).str.strip().str.lower()
    titles_df['category'] = titles_df.get('category', '').fillna('').astype(str).str.strip()

    return keywords_df, titles_df

def main():
    file_path = "C:/Users/ASUS/Downloads/Data/Raw/DatasetFPT.xlsx"
    keyword_sheet = "Clean Keyword"
    title_sheet = "Clean Title"
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)

    df_keywords, df_titles = load_clean_data(file_path, keyword_sheet, title_sheet)

    df_keywords = df_keywords.head(100)
    df_titles = df_titles.head(100)


    df_keywords, df_titles = prepare_data(df_keywords, df_titles)

    print("Predicting sub-categories for keywords...")
    model_path = "Modeling/model_output"  # Local path to fine-tuned BERT model
    df_keywords = predict_lables(df_keywords, model_path)

    print("Matching keywords to titles using Option B...")
    matcher = KeywordMatcher()  # PhoBERT still used for similarity unless replaced
    matches_df = matcher.match_keywords_to_titles(df_keywords, df_titles, top_k=1, threshold=0.6)

    print("Aggregating results...")
    summary_df = aggregate_results(matches_df)

    print("Saving results to Excel...")
    save_to_excel(matches_df, file_path, sheet_name="Matches")
    save_to_excel(summary_df, file_path, sheet_name="Summary")
    print("Done.")

if __name__ == "__main__":
    main()
