import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

parser = argparse.ArgumentParser(description="Train a DistilBERT model.")
parser.add_argument("--task", type=str, required=True, choices=["classification"], help="The task to train on.")
args = parser.parse_args()

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("glue", "sst2")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

train_dataset = dataset["train"].select(range(200)).map(preprocess, batched=True)
eval_dataset = dataset["validation"].select(range(100)).map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="distilbert-finetuned-classification",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_dir="./logs",
    seed=42
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained("distilbert-finetuned-classification")
