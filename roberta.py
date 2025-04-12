import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
from seqeval.metrics import classification_report

# Constants
roberta_version = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 3
batch_size = 4
learning_rate = 1e-5

def load_and_prepare_dataset():
    """Load and preprocess the dataset with proper label alignment"""
    tokenizer = RobertaTokenizerFast.from_pretrained(roberta_version, add_prefix_space=True)
    dataset = load_from_disk('conll2003')
    
    def align_labels(example):
        tokenized = tokenizer(
            example['tokens'],
            truncation=True,
            padding='max_length',
            is_split_into_words=True,
            max_length=128
        )
        
        # Get word IDs for each token
        word_ids = tokenized.word_ids()
        
        # Align labels with tokens
        labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:  # Special tokens
                labels.append(-100)
            elif word_id != current_word:  # New word
                current_word = word_id
                labels.append(example['ner_tags'][word_id])
            else:  # Subword token
                labels.append(-100)
        
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(align_labels, batched=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset, tokenizer

def initialize_model(dataset):
    """Initialize model with correct label mappings"""
    label_names = dataset['train'].features['ner_tags'].feature.names
    num_labels = len(label_names)
    
    model = RobertaForTokenClassification.from_pretrained(
        roberta_version,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)}
    )
    return model, num_labels

def train_model(model, dataset, optimizer, n_epochs, batch_size):
    """Improved training loop with proper loss calculation"""
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    train_loss = []
    
    model.to(device)
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_epoch_loss)
    
    return train_loss

def evaluate_model(model, dataset):
    """Proper evaluation with seqeval metrics"""
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        
        # Remove ignored labels (padding)
        for i in range(len(preds)):
            mask = labels[i] != -100
            valid_preds = preds[i][mask]
            valid_labels = labels[i][mask]
            
            all_preds.append([model.config.id2label[p] for p in valid_preds])
            all_labels.append([model.config.id2label[l] for l in valid_labels])
    
    return classification_report(all_labels, all_preds)

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    dataset, tokenizer = load_and_prepare_dataset()
    model, num_labels = initialize_model(dataset)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    model.train()
    train_loss = train_model(model, dataset, optimizer, n_epochs, batch_size)
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training Loss')
    plt.show()
    
    # Evaluate
    report = evaluate_model(model, dataset)
    print("Classification Report:")
    print(report)