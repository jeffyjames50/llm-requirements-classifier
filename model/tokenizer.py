from transformers import DistilBertTokenizerFast

def get_tokenizer(model_name="distilbert-base-uncased"):
    return DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["statement"], padding=True, truncation=True), batched=True)
