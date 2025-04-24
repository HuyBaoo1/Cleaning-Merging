import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from pyvi import ViTokenizer
from spellchecker import SpellChecker

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize reusable components
lemmatizer = WordNetLemmatizer()
spell_checker = SpellChecker()
stop_words = set(stopwords.words('english'))

# Vietnamese stopwords
vietnamese_stopwords = {"và", "là", "có", "của", "cho", "với", "được", "trong",
                        "này", "một", "đó", "không", "cũng", "nhưng", "nếu",
                        "thì", "để", "các", "điều"}
stop_words.update(vietnamese_stopwords)

# Spelling correction function
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

# Normalization function
def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    vietnamese_chars = r'àáạãảâấầậẫẩăắằặẵẳèéẹẽẻêếềệễểìíịĩỉòóọõỏôốồộỗổơớờợỡởùúụũủưứừựữửỳýỵỹỷđ'
    try:
        text = str(text).lower().strip()
        text = re.sub(fr'[^\w\s{vietnamese_chars}]', '', text)
        text = correct_spelling(text, spell_checker)
        text = ViTokenizer.tokenize(text)
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(words)
    except Exception as e:
        print(f"Text error: {text}. Error: {str(e)}")
        return ""

# Function to clean titles
def clean_titles(titles_df):
    """
    Function to clean and preprocess titles.
    """
    if 'title' not in titles_df.columns or 'title_en' not in titles_df.columns:
        raise ValueError("The DataFrame must contain 'title' and 'title_en' columns.")

    # Combine and clean titles
    titles_df['normalized_title'] = (titles_df['title'].fillna('') + ' ' + titles_df['title_en'].fillna('')).apply(normalize_text)

    # Remove rows with empty normalized titles
    titles_df = titles_df[titles_df['normalized_title'].str.strip().astype(bool)]

    return titles_df

# Main function for standalone execution
def main():
    try:
        file_path = "C:/Users/ASUS/Downloads/Data/Raw/DatasetFPT.xlsx"

        # Load data
        df = pd.read_excel(file_path, sheet_name=5)

        required_columns = {'title', 'title_en', 'category', 'sub_category'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")

        # Clean titles
        df = clean_titles(df)

        # Save to Excel
        df.to_excel(file_path, sheet_name='Clean Title', index=False)
        print(f"Saved {len(df)} cleaned titles to Excel.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
