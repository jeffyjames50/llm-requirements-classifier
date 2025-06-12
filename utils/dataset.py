import pandas as pd
from datasets import Dataset

def load_and_prepare_dataset(path: str):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("label", "labels")
    return dataset
