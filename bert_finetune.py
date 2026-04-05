# ============================================================
# Aspect-Based Sentiment Analysis — BERT Fine-Tuning
# ============================================================
# What are we doing here?
#
#   BERT (Bidirectional Encoder Representations from Transformers)
#   is a pre-trained language model. It already "understands"
#   English from reading billions of sentences.
#
#   Fine-tuning = we take that pre-trained BERT and train it
#   a little more on OUR specific task: given a sentence and
#   an aspect term, predict the sentiment (positive/neutral/negative).
#
#   Input  : "The battery life is great but the screen is bad"
#              aspect = "battery life"
#   Output : POSITIVE
#
#   Input format for BERT:
#     [CLS] sentence [SEP] aspect_term [SEP]
#     BERT reads both together so it understands context.
#
# ============================================================
# Install (run once):
#   pip install torch transformers datasets scikit-learn
#               matplotlib seaborn tqdm
# ============================================================

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm


# ============================================================
# STEP 1: Configuration — all settings in one place
# ============================================================
# Change these values to experiment (ablation studies).

CONFIG = {
    'model_name':    'bert-base-uncased',  # pre-trained model to start from
    'max_length':    128,       # max tokens per input (sentence + aspect)
    'batch_size':    16,        # how many samples per training step
    'num_epochs':    4,         # how many times to loop through all data
    'learning_rate': 2e-5,      # how fast to update weights (small = stable)
    'warmup_ratio':  0.1,       # fraction of steps used to warm up LR
    'weight_decay':  0.01,      # regularization (prevents overfitting)
    'seed':          42,        # for reproducibility
    'num_labels':    3,         # positive, neutral, negative
    'output_dir':    'bert_absa_model',
}

# Label mappings
LABEL2ID = {'positive': 0, 'neutral': 1, 'negative': 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# Set random seeds so results are reproducible
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# Use GPU if available (much faster), otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# STEP 2: Load and parse SemEval data
# ============================================================

def parse_semeval_for_bert(filepath):
    """
    Reads SemEval XML. Returns a list of dicts:
      { 'text': sentence, 'aspect': term, 'polarity': label }
    Each dict = one training example.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    samples = []

    for sentence in root.findall('.//sentence'):
        text_el = sentence.find('text')
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()

        for container_tag in ['aspectTerms', 'Opinions']:
            container = sentence.find(container_tag)
            if container is not None:
                child_tag = 'aspectTerm' if container_tag == 'aspectTerms' else 'Opinion'
                for asp in container.findall(child_tag):
                    term     = asp.get('term', asp.get('target', ''))
                    polarity = asp.get('polarity', asp.get('sentiment', ''))

                    # Only keep the 3 main classes; skip 'conflict'
                    if term and term.upper() != 'NULL' and polarity in LABEL2ID:
                        samples.append({
                            'text':     text,
                            'aspect':   term,
                            'polarity': polarity,
                        })
    return samples


# Demo data if no XML files found
DEMO_DATA = [
    {'text': 'The battery life is excellent.',          'aspect': 'battery life', 'polarity': 'positive'},
    {'text': 'Battery life is excellent but screen bad.','aspect': 'screen',      'polarity': 'negative'},
    {'text': 'The food was amazing!',                   'aspect': 'food',         'polarity': 'positive'},
    {'text': 'Service was absolutely terrible.',        'aspect': 'service',      'polarity': 'negative'},
    {'text': 'The pasta was okay, nothing special.',    'aspect': 'pasta',        'polarity': 'neutral'},
    {'text': 'Ambiance was great, food was average.',   'aspect': 'ambiance',     'polarity': 'positive'},
    {'text': 'Ambiance was great, food was average.',   'aspect': 'food',         'polarity': 'neutral'},
    {'text': 'Keyboard feels mushy and unresponsive.',  'aspect': 'keyboard',     'polarity': 'negative'},
    {'text': 'Display is razor sharp and vivid.',       'aspect': 'display',      'polarity': 'positive'},
    {'text': 'The wifi range is decent.',               'aspect': 'wifi',         'polarity': 'neutral'},
    {'text': 'Pizza was cold and tasteless.',           'aspect': 'pizza',        'polarity': 'negative'},
    {'text': 'Staff were friendly and attentive.',      'aspect': 'staff',        'polarity': 'positive'},
    {'text': 'Coffee was just average.',                'aspect': 'coffee',       'polarity': 'neutral'},
    {'text': 'The screen resolution is outstanding.',   'aspect': 'screen',       'polarity': 'positive'},
    {'text': 'Sound quality is mediocre at best.',      'aspect': 'sound quality','polarity': 'neutral'},
    {'text': 'Portions are generous and delicious.',    'aspect': 'portions',     'polarity': 'positive'},
]

DATA_FILES = [
    'data/Restaurants_Train_v2.xml',
    'data/Laptop_Train_v2.xml',
]

all_samples = []
for fpath in DATA_FILES:
    if os.path.exists(fpath):
        parsed = parse_semeval_for_bert(fpath)
        all_samples.extend(parsed)
        print(f"✅ Loaded {len(parsed)} samples from {fpath}")

if not all_samples:
    print("📌 No XML files found — using demo data (16 samples).")
    print("   Place SemEval XML files in data/ for real training.\n")
    all_samples = DEMO_DATA

print(f"\nTotal samples : {len(all_samples)}")
print(f"Label distribution: {Counter(s['polarity'] for s in all_samples)}")


# ============================================================
# STEP 3: Split into train / validation / test sets
# ============================================================
# Standard split: 80% train, 10% validation, 10% test
# Stratified = keeps the same class balance in each split

df = pd.DataFrame(all_samples)

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=CONFIG['seed'],
    stratify=df['polarity']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=CONFIG['seed'],
)

print(f"\nTrain : {len(train_df)} samples")
print(f"Val   : {len(val_df)} samples")
print(f"Test  : {len(test_df)} samples")


# ============================================================
# STEP 4: PyTorch Dataset class
# ============================================================
# PyTorch needs data wrapped in a Dataset class.
# This class tokenizes each (sentence, aspect) pair for BERT.
#
# BERT input format:
#   [CLS] I loved the food [SEP] food [SEP]
#   Token IDs + Attention mask + Token type IDs

class ABSADataset(Dataset):
    """
    Converts (text, aspect, polarity) triples into
    BERT-ready tensors.
    """
    def __init__(self, data, tokenizer, max_length):
        self.data       = data.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        text   = row['text']
        aspect = row['aspect']
        label  = LABEL2ID[row['polarity']]

        # Tokenize: sentence paired with aspect term
        # The tokenizer handles [CLS], [SEP], padding, truncation
        encoding = self.tokenizer(
            text,
            aspect,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids',
                              torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# STEP 5: Load tokenizer and model
# ============================================================

print(f"\n📥 Loading tokenizer and model: {CONFIG['model_name']}")
tokenizer = BertTokenizerFast.from_pretrained(CONFIG['model_name'])

model = BertForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=CONFIG['num_labels'],
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters     : {total_params:,}")
print(f"   Trainable parameters : {trainable_params:,}")

# Create datasets and dataloaders
train_dataset = ABSADataset(train_df, tokenizer, CONFIG['max_length'])
val_dataset   = ABSADataset(val_df,   tokenizer, CONFIG['max_length'])
test_dataset  = ABSADataset(test_df,  tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'], shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'], shuffle=False)


# ============================================================
# STEP 6: Optimizer and learning rate schedule
# ============================================================
# AdamW = Adam optimizer with weight decay (standard for BERT)
# LR Schedule = warmup then linear decay
#   - Warmup: start with tiny LR, ramp up (prevents early instability)
#   - Decay:  slowly reduce LR as training progresses

optimizer = AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
)

total_steps  = len(train_loader) * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"\n📊 Training schedule:")
print(f"   Total steps   : {total_steps}")
print(f"   Warmup steps  : {warmup_steps}")
print(f"   Steps per epoch: {len(train_loader)}")


# ============================================================
# STEP 7: Training and evaluation functions
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, device):
    """One full pass through the training data."""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()   # clear old gradients

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()                          # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()                         # update weights
        scheduler.step()                         # update learning rate

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate model on validation or test data."""
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            preds       = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# STEP 8: Training loop
# ============================================================

print(f"\n🚀 Starting training for {CONFIG['num_epochs']} epochs...\n")

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc   = 0
best_model_path = os.path.join(CONFIG['output_dir'], 'best_model')
os.makedirs(CONFIG['output_dir'], exist_ok=True)

for epoch in range(1, CONFIG['num_epochs'] + 1):
    print(f"Epoch {epoch}/{CONFIG['num_epochs']}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    val_loss,   val_acc, _, _ = evaluate(model, val_loader, DEVICE)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")

    # Save the best model (by validation accuracy)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")
    print()


# ============================================================
# STEP 9: Final evaluation on test set
# ============================================================

print("=" * 55)
print("FINAL TEST SET EVALUATION")
print("=" * 55)

# Load the best saved model for testing
best_model = BertForSequenceClassification.from_pretrained(best_model_path)
best_model.to(DEVICE)

_, test_acc, test_preds, test_labels = evaluate(best_model, test_loader, DEVICE)

print(f"\nTest Accuracy: {test_acc:.4f}\n")
print("Per-class Report:")
print(classification_report(
    test_labels, test_preds,
    target_names=['positive', 'neutral', 'negative']
))


# ============================================================
# STEP 10: Visualizations
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('BERT Fine-Tuning Results', fontsize=14, fontweight='bold')

epochs_range = range(1, CONFIG['num_epochs'] + 1)

# Plot 1: Loss curves
ax = axes[0]
ax.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
ax.plot(epochs_range, history['val_loss'],   'r-o', label='Val Loss')
ax.set_title('Loss Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Accuracy curves
ax = axes[1]
ax.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
ax.plot(epochs_range, history['val_acc'],   'r-o', label='Val Acc')
ax.set_title('Accuracy Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Confusion matrix
ax = axes[2]
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(
    cm, annot=True, fmt='d', ax=ax,
    xticklabels=['positive', 'neutral', 'negative'],
    yticklabels=['positive', 'neutral', 'negative'],
    cmap='Blues',
)
ax.set_title('Confusion Matrix (Test Set)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

plt.tight_layout()
plt.savefig('bert_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training plots saved to 'bert_training_results.png'")


# ============================================================
# STEP 11: Inference — predict on new reviews
# ============================================================

def predict_sentiment(text, aspect, model, tokenizer, device):
    """
    Given a sentence and an aspect term, return predicted sentiment.
    This is what your Streamlit app will call.
    """
    model.eval()
    encoding = tokenizer(
        text,
        aspect,
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            token_type_ids=encoding.get('token_type_ids',
                           torch.zeros(1, CONFIG['max_length'], dtype=torch.long)).to(device),
        )

    probs     = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id   = probs.argmax().item()
    label     = ID2LABEL[pred_id]
    confidence = probs[pred_id].item()

    return {
        'sentiment':   label,
        'confidence':  confidence,
        'scores': {
            'positive': probs[0].item(),
            'neutral':  probs[1].item(),
            'negative': probs[2].item(),
        }
    }


# Test the inference function
print("\n--- Inference Examples ---")
TEST_EXAMPLES = [
    ('The battery life lasted all day — very impressed.', 'battery life'),
    ('Service was painfully slow and rude.',              'service'),
    ('The pasta was nothing special, just average.',      'pasta'),
    ('Screen resolution is absolutely stunning.',         'screen'),
]

for text, aspect in TEST_EXAMPLES:
    result = predict_sentiment(text, aspect, best_model, tokenizer, DEVICE)
    emoji  = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}[result['sentiment']]
    print(f"\n  Text   : {text}")
    print(f"  Aspect : {aspect}")
    print(f"  Result : {emoji} {result['sentiment'].upper()} ({result['confidence']:.1%} confidence)")
    print(f"  Scores : pos={result['scores']['positive']:.2f}  "
          f"neu={result['scores']['neutral']:.2f}  "
          f"neg={result['scores']['negative']:.2f}")

print("\n🎉 BERT fine-tuning complete!")
print("   Model saved to:", best_model_path)
print("   Next step: Attention visualization with BertViz.")
