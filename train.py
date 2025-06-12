import os
import json
import csv
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments
from model.classifier import load_model
from model.tokenizer import get_tokenizer, tokenize_dataset
from utils.dataset import load_and_prepare_dataset
from utils.metrics import compute_metrics


def log_metrics_to_csv(logs, filename="results/metrics_log.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=logs.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(logs)


def plot_metrics_from_csv(csv_path="results/metrics_log.csv", output_path="results/training_plot.png"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    if "eval_loss" in df:
        plt.plot(df["epoch"], df["eval_loss"], marker='o', label="Eval Loss")
    if "eval_accuracy" in df:
        plt.plot(df["epoch"], df["eval_accuracy"], marker='o', label="Eval Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Evaluation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
        save_total_limit=1,
        save_strategy="no",  # No intermediate checkpoint saving
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final evaluation
    results = trainer.evaluate()

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/eval_report.json", "w") as f:
        json.dump(results, f, indent=4)

    # Log to CSV
    log_metrics_to_csv(results)

    # Plot metrics
    plot_metrics_from_csv()

    # Save final model and tokenizer
    model.save_pretrained("requirement_classifier")
    tokenizer.save_pretrained("requirement_classifier")
    print("Training complete. Model saved to requirement_classifier/")
    print("Evaluation metrics saved to results/eval_report.json and results/metrics_log.csv")
    print("Training plot saved to results/training_plot.png")


if __name__ == "__main__":
    main()
