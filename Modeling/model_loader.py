class ModelLoader:
    def __init__(self, model_path="distilbert-base-multilingual-cased", device=None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        return self.tokenizer, self.model

    def save_model(self, model, tokenizer):
        print(f"Saving model to {self.model_path}...")
        tokenizer.save_pretrained(self.model_path)
        model.save_pretrained(self.model_path)
        print("Model saved successfully.")
