def train_classifier(
    titles_df,
    model_name="distilbert-base-multilingual-cased",
    output_dir="./Modeling/model_output"
):
    try:
        print(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")


    label_encoder = LabelEncoder()
    titles_df["label"] = label_encoder.fit_transform(titles_df["sub_category"])

    print("Preparing dataset...")
    dataset = Dataset.from_pandas(titles_df[["normalized_title", "label"]].rename(columns={"normalized_title": "text"}))

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    try:
        print("Loading pretrained model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_encoder.classes_),
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        num_train_epochs=5,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1,
        save_strategy="epoch",
        logging_steps=1000,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    print("Starting training...")
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    print("Saving model, tokenizer and label encoder...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    print("Training complete.")
    return output_dir, label_encoder
