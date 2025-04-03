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
NUM_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MAX_LENGTH = 128
WEIGHT_MAP = {
    "O": 1.0,
    "B_LOC": 1.0,
    'I_LOC': 1.0,
    'B_MISC': 1.0,
    'I_MISC': 1.0,
    'B_ORG': 1.0,
    'I_ORG': 1.0,
    'B_PER': 1.0,
    'I_PER': 1.0
}
LOSS_FUNCTION = "cross_entropy" # Options: "cross_entropy", "focal_loss", "dice_loss"

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Compute softmax
        BCE_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # Focal loss formula
        return F_loss.mean()  # Return the mean loss
    
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        return 1 - dice_score.mean()  # Return the mean loss
    
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

def tokenize_examples(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    return tokenized

def create_labels(tokenized, examples):
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
        lambda x: create_labels(tokenize_examples(x, tokenizer), x),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    
    return tokenized_ds, label_names, tokenizer

def create_data_loaders(tokenized_ds, tokenizer):
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return {
        "train": DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, pin_memory=True),
        "validation": DataLoader(tokenized_ds["validation"], batch_size=BATCH_SIZE, collate_fn=collator, pin_memory=True),
        "test": DataLoader(tokenized_ds["test"], batch_size=BATCH_SIZE, collate_fn=collator, pin_memory=True),
    }

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

def initialize_optimizer_and_scheduler(model, train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def get_loss_function(loss_type, weights=None):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
    elif loss_type == "focal_loss":
        return FocalLoss(alpha=1.0, gamma=2.0, ignore_index=-100)
    elif loss_type == "dice_loss":
        return DiceLoss(ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
def train_model(model, train_loader, val_loader, label_names, weights, training_stats):
    print("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer, scheduler = initialize_optimizer_and_scheduler(model, train_loader)
    loss_fn = get_loss_function(LOSS_FUNCTION, weights)
    best_f1 = 0

    for epoch in range(NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        training_stats['loss'].append(loss)
        report = validate_model(model, val_loader, label_names, device)
        val_f1 = report["micro avg"]["f1-score"]
        recall = report["micro avg"]["recall"]
        precision = report["micro avg"]["precision"]

        training_stats['recalls'].append(recall)
        training_stats['precisions'].append(precision)
        training_stats['f1_scores'].append(val_f1)
       
        if val_f1 > best_f1:
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
        
    return epoch_loss / len(pbar)

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
    print(classification_report(all_labels, all_preds))
    return report

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

def plot_results(training_stats):
    x = np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS)
    plt.figure(figsize=(10, 5))
    plt.plot(x, training_stats['recalls'], label='Recall')
    plt.plot(x, training_stats['precisions'], label='Precision')
    plt.plot(x, training_stats['f1_scores'], label='F1-Score')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Metrics')
    plt.title('Evaluation Metrics Over Epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x, training_stats['loss'], label='Loss', color='red')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    tokenized_ds, label_names, tokenizer = process_data()
    loaders = create_data_loaders(tokenized_ds, tokenizer)
    model = NERModel(num_labels=len(label_names))
    weights = list(WEIGHT_MAP.values())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_stats = {
        'loss': [],
        'recalls': [],
        'precisions': [],
        'f1_scores': []
    }
    
    train_model(model, loaders['train'], loaders['validation'], label_names, weights, training_stats)
    evaluate_model(model, loaders['test'], label_names, device)
    plot_results(training_stats)