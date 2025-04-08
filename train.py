from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchcrf import CRF

# Configuration
MODEL_NAME = "bert-base-cased"
DATASET_NAME = "conll2003"
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
CRF_LEARNING_RATE = 1e-3
MAX_LENGTH = 128

class ClassLabel(Enum):
    O = 0
    B_LOC = 1
    I_LOC = 2
    B_MISC = 3
    I_MISC = 4
    B_ORG = 5
    I_ORG = 6
    B_PER = 7
    I_PER = 8

# Model Definition
class NERModel(nn.Module):
    def __init__(self, num_labels=9):
        super().__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits
        
        if labels is not None:
            mask = (labels != -100).float()  # Create mask for CRF
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return logits, loss
        return logits

    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits
        return self.crf.decode(logits)

# Data Processing
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            new_labels.append(-100)
    return new_labels

def load_and_tokenize_data():
    dataset = load_dataset(DATASET_NAME, trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    return dataset, label_names

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def prepare_data():
    dataset, label_names = load_and_tokenize_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_ds = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    
    return tokenized_ds, label_names, tokenizer

def create_data_loaders(tokenized_ds, tokenizer):
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return {
        "train": DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator),
        "validation": DataLoader(tokenized_ds["validation"], batch_size=BATCH_SIZE, collate_fn=collator),
        "test": DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE, collate_fn=collator),
    }

# Training Utilities
def get_optimizer(model):
    bert_params = []
    crf_params = []
    for name, param in model.named_parameters():
        if "crf" in name:
            crf_params.append(param)
        else:
            bert_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": LEARNING_RATE},
        {"params": crf_params, "lr": CRF_LEARNING_RATE}
    ])
    return optimizer

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        _, loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"]
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, data_loader, label_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"]
            )
            labels = batch["labels"].cpu().numpy()
            
            for i in range(len(preds)):
                valid = labels[i] != -100
                all_preds.append([label_names[p] for p in preds[i][valid]])
                all_labels.append([label_names[l] for l in labels[i][valid]])
    
    report = classification_report(all_labels, all_preds, output_dict=True)
    print(classification_report(all_labels, all_preds))
    return report

# Main Training Loop
def main():
    # Prepare data
    tokenized_ds, label_names, tokenizer = prepare_data()
    loaders = create_data_loaders(tokenized_ds, tokenizer)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NERModel(num_labels=len(label_names)).to(device)
    
    # Training setup
    optimizer = get_optimizer(model)
    num_training_steps = len(loaders["train"]) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, loaders["train"], optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_report = evaluate(model, loaders["validation"], label_names, device)
        val_f1 = val_report["micro avg"]["f1-score"]
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model!")
    
    # Final evaluation
    print("\nTesting best model...")
    model.load_state_dict(torch.load("best_model.pt"))
    test_report = evaluate(model, loaders["test"], label_names, device)
    print(f"Test F1: {test_report['micro avg']['f1-score']:.4f}")

if __name__ == "__main__":
    main()