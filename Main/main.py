import sys
import os
import pandas as pd

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Matching.matcher import KeywordMatcher
from Matching.result_aggregator import aggregate_results
from Output.saver import save_to_excel
from Modeling.model_predictor import predict_lables  

def main():
    file_path = "/content/DatasetFPT.xlsx"
    sheet_name = "Clean Title"
    keyword_sheet = "Clean Keyword"

    if not os.path.exists(file_path):
        print(f"Error: The file was not found at {file_path}.")
        print("Please ensure the file is uploaded to the /content/ directory or the path is correct.")
        return
    if not os.path.isfile(file_path):
        print(f"Error: The path {file_path} exists but is not a file.")
        return

    # Load titles
    try:
        # Specify the engine explicitly
        titles_df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        keywords_df = pd.read_excel(file_path, sheet_name=keyword_sheet, engine="openpyxl")
        print("File loaded successfully.")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    titles_df = titles_df[titles_df["category"].str.lower() == "anime"]


    titles_df["title"] = titles_df["title"].fillna("").astype(str)
    titles_df["title_en"] = titles_df[ "title_en"].fillna("").astype(str)
    titles_df["normalized_title"] = titles_df["normalized_title"].fillna("").astype(str)
    titles_df["sub_category"] = titles_df["sub_category"].fillna("").astype(str)
    keywords_df["normalized_keywords"] = keywords_df["normalized_keywords"].fillna("").astype(str)


    # Train classifier and save to model_output
    print("Training classifier...")
    output_path, label_encoder = train_classifier(titles_df)
    print(f"Loading saved model components from {output_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        model = AutoModelForSequenceClassification.from_pretrained(output_path)
        label_encoder = joblib.load(os.path.join(output_path, "label_encoder.pkl"))
        label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print("Saved model components loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading saved model components: {e}. Please ensure the model was trained and saved correctly to {output_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading saved model components: {e}")
        return


    print("Predicting labels for titles...")
    #titles_df_for_prediction = titles_df.copy()
    #titles_df_for_prediction = titles_df_for_prediction.rename(columns={"normalized_title": "normalized_keywords"})
    #predicted_df = predict_lables(titles_df_for_prediction, tokenizer, model, label_map) # Correctly pass all arguments

    sample_df = keywords_df.sample(n=10, random_state=42).copy()

    predicted_sample = predict_lables(sample_df, tokenizer, model, label_map)
    anime_subcategories = titles_df["sub_category"].str.lower().unique()
    predicted_sample = predicted_sample[
    predicted_sample["pre_sub_category"].str.lower().isin(anime_subcategories)]


    print("Prediction for titles completed.")
    print("Sample predictions:")
    display(predicted_sample.head())

    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main()
