import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description="Evaluate a fine-tuned DistilBERT model.")
parser.add_argument("--task", type=str, required=True,
                    choices=["classification", "nli"],
                    help="The evaluation task.")
parser.add_argument("--model_dir", type=str, required=True,
                    help="Path to your saved model directory.")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

if args.task == "classification":
    dataset = load_dataset("glue", "sst2", split="validation").select(range(200))
    dataset = dataset.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length"), batched=True)
    labels = dataset["label"]
elif args.task == "nli":
    dataset = load_dataset("snli", split="validation")
    dataset = dataset.filter(lambda x: x["label"] != -1).select(range(200))
    dataset = dataset.map(lambda e: tokenizer(e["premise"], e["hypothesis"], truncation=True, padding="max_length"), batched=True)
    labels = dataset["label"]

dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

all_preds = []

model.eval()
with torch.no_grad():
    for batch in loader:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())

accuracy = (np.array(all_preds) == np.array(labels)).mean()
print(f"Accuracy on {args.task} validation set: {accuracy:.2%}")
