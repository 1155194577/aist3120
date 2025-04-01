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

class classLabel(Enum):
    O = 0
    B_LOC = 1
    I_LOC = 2
    B_MISC = 3
    I_MISC = 4
    B_ORG = 5
    I_ORG = 6
    B_PER = 7
    I_PER = 8
    
# Configuration
MODEL_NAME = "bert-base-cased"
DATASET_NAME = "conll2003"
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

# Helper function for label alignment
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

# Model definition
class NERModel(nn.Module):
    def __init__(self, num_labels=9):
        super().__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits

# Data processing functions
def load_and_tokenize_data():
    dataset = load_dataset(DATASET_NAME, trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    return dataset, label_names

def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    
    tokenized["labels"] = new_labels
    return tokenized

def process_data():
    dataset, label_names = load_and_tokenize_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_ds = dataset.map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    
    return tokenized_ds, label_names, tokenizer

# Data loader creation
def create_loader(tokenized_ds, tokenizer):
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return {
        "train": DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, pin_memory=True),
        "validation": DataLoader(tokenized_ds["validation"], batch_size=BATCH_SIZE, collate_fn=collator, pin_memory=True),
        "test": DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE, collate_fn=collator, pin_memory=True),
    }

# Training loop
def train_model(model, train_loader, val_loader, label_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_f1 = 0
    
    for epoch in range(NUM_EPOCHS):
        train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        val_f1 = validate_model(model, val_loader, label_names, device)
        
        if val_f1 > best_f1:
            save_best_model(model)
            best_f1 = val_f1

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    epoch_loss = 0
    
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as pbar:
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), batch["labels"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

def validate_model(model, val_loader, label_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
            for i in range(len(preds)):
                valid = labels[i] != -100
                all_preds.append([label_names[p] for p in preds[i][valid]])
                all_labels.append([label_names[l] for l in labels[i][valid]])
    
    report = classification_report(all_labels, all_preds, output_dict=True)
    val_f1 = report["micro avg"]["f1-score"]
    print(f"\nValidation F1: {val_f1:.4f}")
    print(classification_report(all_labels, all_preds))
    return val_f1

def save_best_model(model):
    model.save_pretrained("best_model")
    print("New best model saved!")

# Evaluation
def evaluate_model(trained_model, test_loader, label_names, device):
    trained_model.eval()
    trained_model.to(device)
    
    all_preds = []
    all_labels = []
    confusion_matrix_data = np.zeros((len(classLabel), len(classLabel)), dtype=int)
    
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", desc="Evaluating") as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]
                
                outputs = trained_model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()

                for i in range(len(preds)):
                    valid = labels[i] != -100
                    all_preds.append([label_names[p] for p in preds[i][valid]])
                    all_labels.append([label_names[l] for l in labels[i][valid]])
                    
                    for x, y in zip(all_preds[-1], all_labels[-1]):
                        confusion_matrix_data[classLabel[y.replace('-', '_')].value, classLabel[x.replace('-', '_')].value] += 1
                
                pbar.set_postfix({'Processed': len(all_preds)})

    report = classification_report(all_labels, all_preds, output_dict=True)
    test_f1 = report["micro avg"]["f1-score"]
    print(f"\nTest F1: {test_f1:.4f}")
    print(classification_report(all_labels, all_preds))
    
    plot_confusion_matrix(confusion_matrix_data, label_names)

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def get_loader(): 
   tokenized_ds, label_names, tokenizer = process_data() 
   return create_loader(tokenized_ds, tokenizer)

# Main execution
if __name__ == "__main__":
    tokenized_ds, label_names, tokenizer = process_data()
    loaders = create_loader(tokenized_ds, tokenizer)
    model = NERModel(num_labels=len(label_names))
    
    # Uncomment to train the model
    # train_model(model, loaders['train'], loaders['validation'], label_names)
    
    evaluate_model(model, loaders['test'], label_names, torch.device("cuda" if torch.cuda.is_available() else "cpu"))