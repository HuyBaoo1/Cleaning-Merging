import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from Matching.matcher import KeywordMatcher
from Matching.result_aggregator import aggregate_results
from Output.saver import save_to_excel

#pip install openpyxl
#pip install transformers

def load_clean_data(file_path, keyword_sheet, title_sheet):
    """
    Load pre-cleaned keywords and titles data from an Excel file.
    """
    keywords_df = pd.read_excel(file_path, sheet_name=keyword_sheet)
    titles_df = pd.read_excel(file_path, sheet_name=title_sheet)
    return keywords_df, titles_df

def main():
    # Define file path and sheet names
    file_path = "C:/Users/ASUS/Downloads/Data/Raw/DatasetFPT.xlsx"
    keyword_sheet = "Clean Keyword"  # Use the pre-cleaned keyword sheet
    title_sheet = "Clean Title"      # Use the pre-cleaned title sheet

    # Verify and create the Output directory if it doesn't exist
    output_dir = "Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load pre-cleaned data
    df_keywords, df_titles = load_clean_data(file_path, keyword_sheet, title_sheet)

    # Preprocess keywords and titles to handle missing values
    df_keywords['normalized_keywords'] = df_keywords['normalized_keywords'].fillna('').astype(str)  # Ensure keywords are strings
    df_titles['combined_title'] = df_titles['normalized_title'].fillna('').astype(str) 
    df_titles['title'] = df_titles['title'].fillna('').astype(str)          # Ensure titles are strings
    df_titles['title_en'] = df_titles['title_en'].fillna('').astype(str)    # Ensure English titles are strings

    # 2. Match keywords to titles
    matcher = KeywordMatcher()
    matches_df = matcher.match_keywords_to_titles(df_keywords, df_titles)

    # 3. Aggregate results by category/sub-category
    summary_df = aggregate_results(matches_df)

    # 4. Save results to the same Excel file in new sheets
    save_to_excel(matches_df, file_path, "Matches")  # Save matches to a new sheet
    save_to_excel(summary_df, file_path, "Summary")  # Save summary

if __name__ == "__main__":
    main()
