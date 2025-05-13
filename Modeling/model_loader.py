import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelLoader:
    def __init__(self, model_name="bert-base-multilingual-cased", device=None):
        selfmodel_path = model_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")

        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
