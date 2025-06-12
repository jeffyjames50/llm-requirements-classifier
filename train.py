from transformers import Trainer, TrainingArguments
from model.classifier import load_model
from model.tokenizer import get_tokenizer, tokenize_dataset
from utils.dataset import load_and_prepare_dataset

def main():
    # Load & prepare dataset
    dataset = load_and_prepare_dataset("data/requirements.csv")
    tokenizer = get_tokenizer()
    dataset = tokenize_dataset(dataset, tokenizer)
    dataset = dataset.train_test_split(test_size=0.2)

    # Load model
    model = load_model()

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.evaluate()

    # Save model
    model.save_pretrained("requirement_classifier")
    tokenizer.save_pretrained("requirement_classifier")
    print("Training complete. Model saved to requirement_classifier/")

if __name__ == "__main__":
    main()
