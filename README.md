# llm-requirements-classifier

A hands-on NLP project that fine-tunes a DistilBERT model to automatically classify software requirement statements into **Functional** or **Non-Functional** categories.

## Project Structure
llm-requirements-classifier/
├── data/
│ └── requirements.csv # Labeled dataset: “statement”, “label” (0=Functional,1=Non-Functional)
│
├── model/
│ ├── classifier.py # load_model() wrapper
│ └── tokenizer.py # get_tokenizer() & tokenize_dataset()
│
├── utils/
│ ├── dataset.py # load_and_prepare_dataset()
│ └── metrics.py # compute_metrics() for accuracy
│
├── train.py # Train, evaluate, log + plot metrics, save model/tokenizer
├── predict.py # Batch inference via HF pipeline + label map
├── app.py # Streamlit demo: text input, predictions & metrics plot
│
├── results/ # ── generated at training time ──
│ ├── eval_report.json # Final evaluation metrics (loss, accuracy…)
│ ├── metrics_log.csv # Epoch-wise metrics log
│ └── training_plot.png # Loss & accuracy curves
│
├── requirement_classifier/ # Saved DistilBERT model & tokenizer
│
├── .gitignore # Ignored files & folders (venv, logs, results…)
├── requirements.txt # Exact package versions for reproducibility
└── README.md # This file

## Features

- **Fine-tuning** of `distilbert-base-uncased` on a small, custom requirements dataset  
- **Custom accuracy metric** via `compute_metrics()`  
- **Epoch-wise logging** to both CSV and JSON, with matplotlib plotting  
- **CLI inference** (`predict.py`) and **interactive demo** (`app.py` via Streamlit)  
- **Clean code structure** for easy extension & maintenance  

## Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/jeffyjames50/llm-requirements-classifier.git
   cd llm-requirements-classifier

2.**Create a Python 3.10 virtual environment**
py -3.10 -m venv venv
.\venv\Scripts\activate

3.**Install dependencies**
pip install -r requirements.txt
## Usage

1. **Train the model**
    ```bash
    python train.py
    ```
    This will generate:
    - `results/eval_report.json`
    - `results/metrics_log.csv`
    - `results/training_plot.png`
    - `requirement_classifier/` (model & tokenizer)

2. **Make Batch Predictions**

python predict.py

-Prints predictions for hard-coded examples.

-Edit examples list in predict.py or adapt to accept file/CLI input.

3. **Launch Streamlit Demo**

streamlit run app.py

“Classify Requirement” tab: paste one or more lines, click Classify.

“Training Metrics” tab: view the stored training_plot.png and raw CSV.
