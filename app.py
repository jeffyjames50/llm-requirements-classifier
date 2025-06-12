import streamlit as st
import pandas as pd
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TextClassificationPipeline
import torch
import os

# Load tokenizer and model once
@st.cache_resource
def load_model_and_tokenizer(path="requirement_classifier"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(path)
    model = DistilBertForSequenceClassification.from_pretrained(path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

label_map = {f"LABEL_{i}": name for i, name in enumerate(["Functional", "Non-Functional"])}

st.title("Requirement Classifier Demo")

# Input text box
user_input = st.text_area("Enter requirement text(s) (one per line):")

if st.button("Classify"):
    if user_input.strip():
        texts = [line.strip() for line in user_input.split("\n") if line.strip()]
        st.subheader("Predictions")
        for text in texts:
            prediction = pipeline(text)
            pred_label = max(prediction[0], key=lambda x: x['score'])['label']
            label = label_map.get(pred_label, pred_label)
            st.write(f"**Requirement:** {text}")
            st.write(f"**Predicted Class:** {label}")
            st.write("---")
    else:
        st.warning("Please enter some text to classify.")

# Show training metrics plot if exists
plot_path = "results/training_plot.png"
if os.path.exists(plot_path):
    st.subheader("Training Metrics")
    image = Image.open(plot_path)
    st.image(image, caption="Training metrics (eval loss & accuracy) over epochs")
else:
    st.info("Training plot not found. Please run training first.")
