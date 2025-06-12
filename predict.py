from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TextClassificationPipeline
import torch

# Load tokenizer and model
model_path = "requirement_classifier"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Use pipeline for easy prediction
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# Label mapping (adjust these labels according to your dataset)
label_map = {f"LABEL_{i}": name for i, name in enumerate(["Functional", "Non-Functional"])}

def predict(texts):
    for text in texts:
        prediction = pipeline(text)
        pred_label = max(prediction[0], key=lambda x: x['score'])['label']
        print(f"\nText: {text}")
        print("Predicted Label:", label_map.get(pred_label, pred_label))

if __name__ == "__main__":
    # Example input texts
    examples = [
        "The system must allow users to reset their password via email.",
        "Add CSS styling to the login page.",
        "Database schema should be optimized for query performance."
    ]
    predict(examples)
