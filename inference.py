import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    label = model.config.id2label.get(prediction, str(prediction))
    return label

parser = argparse.ArgumentParser(description="Run inference with your fine-tuned DistilBERT model.")
parser.add_argument("--task", type=str, choices=["classification"], required=True, help="Task to run inference on.")
parser.add_argument("--model_dir", type=str, required=True, help="Relative or absolute path to model directory.")
parser.add_argument("--text", type=str, help="Input text to classify.")

args = parser.parse_args()

# Ensure the model directory is interpreted as a local folder
model_path = Path(args.model_dir).resolve()

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

if args.task == "classification":
    if not args.text:
        raise ValueError("Please provide --text for classification.")
    result = classify(args.text, model, tokenizer)
    print(f"\nInput: {args.text}\nPrediction: {result}")
