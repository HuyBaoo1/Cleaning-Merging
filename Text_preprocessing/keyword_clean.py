import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from pyvi import ViTokenizer
from spellchecker import SpellChecker
from gensim.models import Word2Vec
from openpyxl import load_workbook

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize reusable components
lemmatizer = WordNetLemmatizer()
spell_checker = SpellChecker()
stop_words = set(stopwords.words('english'))
stop_words.update({
    "và", "là", "có", "của", "cho", "với", "được", "trong",
    "này", "một", "đó", "không", "cũng", "nhưng", "nếu",
    "thì", "để", "các", "điều"
})

def correct_spelling(text, spell_checker, max_word_length=20):
    if not text or pd.isna(text):
        return text
    words = text.split()
    corrected_words = []
    for word in words:
        if len(word) > max_word_length or word in spell_checker or re.search(r'\d', word):
            corrected_words.append(word)
        else:
            corrected_words.append(spell_checker.correction(word) or word)
    return ' '.join(corrected_words)

def normalize_text(text):
    vietnamese_chars = r'àáạãảâấầậẫẩăắằặẵẳèéẹẽẻêếềệễểìíịĩỉòóọõỏôốồộỗổơớờợỡởùúụũủưứừựữửỳýỵỹỷđ'
    if not text or pd.isna(text): 
        return ""
    try:
        text = str(text).lower().strip()
        text = re.sub(fr'[^\w\s{vietnamese_chars}]', '', text)
        text = correct_spelling(text, spell_checker)
        text = ViTokenizer.tokenize(text)
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error processing text: {text}. Error: {str(e)}")
        return ""

def clean_keywords(keywords_df):
    """
    Function to clean and preprocess keywords.
    """
    # Normalize keywords
    keywords_df['normalized_keywords'] = keywords_df['keyword'].astype(str).apply(normalize_text)

    # Remove empty or invalid keywords
    keywords_df = keywords_df[keywords_df['normalized_keywords'].str.strip().astype(bool)]
    keywords_df = keywords_df[~keywords_df['keyword'].astype(str).str.match(r'^\d{9,}$', na=False)]
    keywords_df = keywords_df[~keywords_df['keyword'].astype(str).str.match(r'(.)\1{3,}', na=False)]
    keywords_df = keywords_df[~keywords_df['keyword'].astype(str).str.contains(r'[!@#$%^&*()_+\-=\[\]{};:\"\\|,.<>/?~`]', na=False)]

    # Remove frequent and rare words
    all_words = ' '.join(keywords_df['normalized_keywords']).split()
    word_freq = pd.Series(all_words).value_counts()
    frequent_words = set(word_freq[word_freq > 0.95 * len(keywords_df)].index)
    rare_words = set(word_freq[word_freq < 5].index)

    keywords_df['normalized_keywords'] = keywords_df['normalized_keywords'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in frequent_words and w not in rare_words and len(w) > 2])
    )

    return keywords_df

# Main function for standalone execution
def main():
    try:
        RAW_PATH = "c:/Users/ASUS/Downloads/DatasetFPT.xlsx"
        df = pd.read_excel(RAW_PATH, sheet_name=2)

        required_columns = {'keyword', 'Searched Count', 'Search-to-watch'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")

        # Clean keywords
        df = clean_keywords(df)

        # Save cleaned keywords to Excel
        save_to_excel_sheet(df, RAW_PATH, sheet_name="keyword_clean")
        print(f"Saved cleaned keywords to sheet 'keyword_clean' in: {RAW_PATH}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()