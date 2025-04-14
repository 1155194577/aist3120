import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

# Constants
roberta_version = 'roberta-base'
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
n_epochs = 5
batch_size = 32
lr = 0.01
OPTIMIZER = "sgd"  # Options: "sgd", "adagrad", "adam"
LOSS_FN = "dice_loss"  # Options: "cross_entropy", "focal_loss", "dice_loss"
torch.manual_seed(42)

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

def tokenize_examples(examples, tokenizer):
    return tokenizer(
        examples['tokens'],
        truncation=True,
        padding='max_length',
        is_split_into_words=True,
        max_length=128
    )

def create_labels(tokenized, examples):
    word_ids = tokenized.word_ids()
    tokenized["labels"] = align_labels_with_tokens(examples['ner_tags'], word_ids)
    return tokenized

def load_and_tokenize_data():
    tokenizer = RobertaTokenizerFast.from_pretrained(roberta_version, add_prefix_space=True)
    dataset = load_from_disk("conll2003")
    label_names = dataset['train'].features['ner_tags'].feature.names
    return dataset, label_names, tokenizer

def process_data():
    dataset, label_names, tokenizer = load_and_tokenize_data()
    tokenized_ds = dataset.map(lambda x: create_labels(tokenize_examples(x, tokenizer), x), batched=False)
    tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_ds, label_names, tokenizer

def create_data_loaders(tokenized_ds):
    loaders = {
        'train': torch.utils.data.DataLoader(tokenized_ds['train'], batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(tokenized_ds['validation'], batch_size=batch_size),
        'test': torch.utils.data.DataLoader(tokenized_ds['test'], batch_size=batch_size)
    }
    return loaders

def initialize_optimizer_and_scheduler(model, train_loader):
    if OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif OPTIMIZER == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, eps=1e-6)
    elif OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-6)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return optimizer, scheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-100, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=-1)
        mask = targets != self.ignore_index
        targets = torch.where(mask, targets, 0)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.shape[-1]).float()
        probs = probs * mask.unsqueeze(-1)
        targets_one_hot = targets_one_hot * mask.unsqueeze(-1)
        intersection = (probs * targets_one_hot).sum(dim=0)
        sum_probs = probs.sum(dim=0)
        sum_targets = targets_one_hot.sum(dim=0)
        dice = (2. * intersection + self.eps) / (sum_probs + sum_targets + self.eps)
        return 1 - dice.mean()

def get_loss_function(loss_type):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=-100)
    elif loss_type == "focal_loss":
        return FocalLoss(ignore_index=-100)
    elif loss_type == "dice_loss":
        return DiceLoss(ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)
        
        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return epoch_loss / len(train_loader)

def validate_model(model, val_loader, label_names, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(val_loader, desc="Validating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        
        for i in range(len(preds)):
            mask = labels[i] != -100
            valid_preds = preds[i][mask]
            valid_labels = labels[i][mask]
            all_preds.append([model.config.id2label[p] for p in valid_preds])
            all_labels.append([model.config.id2label[l] for l in valid_labels])
    return f1_score(all_labels, all_preds)

def train_model(model, train_loader, val_loader, label_names, training_stats):
    model.to(device)
    optimizer, scheduler = initialize_optimizer_and_scheduler(model, train_loader)
    loss_fn = get_loss_function(LOSS_FN)
    for epoch in range(n_epochs):
        epoch_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        val_f1 = validate_model(model, val_loader, label_names, device)
        training_stats['loss'].append(epoch_loss)
        training_stats['validation_f1_scores'].append(val_f1)
        print(f"Epoch {epoch+1} Loss: {epoch_loss}, Validation F1: {val_f1}")

def plot_results(training_stats):
    plt.figure(figsize=(10, 5))
    plt.plot(training_stats['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(training_stats['validation_f1_scores'], label='Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('f1.png')
    plt.close()

if __name__ == "__main__":
    tokenized_ds, label_names, tokenizer = process_data()
    loaders = create_data_loaders(tokenized_ds)
    model = RobertaForTokenClassification.from_pretrained(
        roberta_version,
        num_labels=len(label_names),
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)}
    )
    training_stats = {'loss': [], 'validation_f1_scores': []}
    train_model(model, loaders['train'], loaders['val'], label_names, training_stats)
    plot_results(training_stats)
    test_f1 = validate_model(model, loaders['test'], label_names, device)
    print(f"Test F1 Score: {test_f1}")