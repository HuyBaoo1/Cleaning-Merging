from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch, pandas as pd, numpy as np
from datasets import Dataset
from model_loader import ModelLoader

def train_classifier(
    titles_df,
    model_name="bert-base-multilingual-cased",
    output_dir="./Modeling/model_output"
):
    try:
        print(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(titles_df["sub_category"])

    print("Preparing dataset...")
    dataset = Dataset.from_pandas(pd.DataFrame({
        "text": titles_df["title"] + " " + titles_df["title_en"],
        "label": labels
    }))

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    try:
        print("Loading model from pretrained...")
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
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
        save_strategy="epoch",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none"  # Avoid warning if not using wandb or other tools
    )

    print("Starting training...")
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete.")
    return output_dir, label_encoder
