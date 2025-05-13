from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch, pandas as pd, numpy as np
from datasets import Dataset
from Modeling.model_loader import ModelLoader

def train_classifier(titles_df, model_name="bert-base-multilingual-cased", output_dir="./Modeling/model_output"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    labels = LabelEncoder().fit_transform(titles_df["sub_category"])

    dataset = Dataset.from_pandas(pd.DataFrame({
        "text": titles_df["title"] + " " + titles_df["title_en"],
        "label": labels
    }))

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(labels)))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
        save_strategy="epoch"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(output_dir)

    model_loader = ModelLoader(model_path=output_dir)
    model_loader.save_model(model, tokenizer)

    return output_dir, label_encoder # Save the label encoder for later use
