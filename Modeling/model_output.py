def main():
    file_path = "/content/DatasetFPT.xlsx"
    sheet_name = "Clean Title"

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
        print("File loaded successfully.")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return


    titles_df["title"] = titles_df["title"].fillna("").astype(str)
    titles_df["title_en"] = titles_df[ "title_en"].fillna("").astype(str)
    titles_df["normalized_title"] = titles_df["normalized_title"].fillna("").astype(str)
    titles_df["sub_category"] = titles_df["sub_category"].fillna("").astype(str)


    # Train classifier and save to model_output
    print("Training classifier...")
    output_path, label_encoder = train_classifier(titles_df)

    predicted_df = predict_lables(titles_df, output_path)

    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main()
