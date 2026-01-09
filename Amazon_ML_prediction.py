
# STEP 1: Install Dependencies

!pip install transformers torch pandas scikit-learn tqdm


# STEP 2: Import Libraries

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# STEP 3: Load Dataset

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(train.head())


# STEP 4: Preprocess Text

train['catalog_content'] = train['catalog_content'].astype(str).str.lower()
test['catalog_content'] = test['catalog_content'].astype(str).str.lower()

# Remove missing prices if any
train = train.dropna(subset=['price']).reset_index(drop=True)
train['log_price'] = np.log1p(train['price'])


# STEP 5: Tokenizer and Dataset Class

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PriceDataset(Dataset):
    def __init__(self, texts, targets=None, max_len=128):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.targets is not None:
            item['targets'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item


# STEP 6: Define Fine-Tuned Model

class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.dropout(pooled)
        return self.regressor(x)


# STEP 7: Prepare Data and Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_texts, val_texts, train_prices, val_prices = train_test_split(
    train['catalog_content'].values,
    train['log_price'].values,
    test_size=0.1,
    random_state=42
)

train_dataset = PriceDataset(train_texts, train_prices)
val_dataset = PriceDataset(val_texts, val_prices)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = PricePredictor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()


# STEP 8: Training Loop

EPOCHS = 8

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device).unsqueeze(1)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")


# STEP 9: SMAPE Metric

def smape(y_true, y_pred):
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


# STEP 10: Validation Evaluation

model.eval()
preds, reals = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].cpu().numpy()
        outputs = model(input_ids, attention_mask).cpu().numpy().flatten()
        preds.extend(outputs)
        reals.extend(targets)

smape_val = smape(np.array(reals), np.array(preds))
print(f"✅ Validation SMAPE: {smape_val:.2f}%")

# STEP 11: Predict Test Data (with sample_id)

test_dataset = PriceDataset(test['catalog_content'].values)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

test_preds = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask).cpu().numpy().flatten()
        test_preds.extend(outputs)

# Reverse log scaling and ensure non-negative prices
test_preds = np.expm1(test_preds)
test['price'] = np.clip(test_preds, a_min=0, a_max=None)

# Ensure sample_id exists in test.csv
if 'sample_id' in test.columns:
    submission = test[['sample_id', 'price']]
else:
    # Fallback: use DataFrame index if sample_id not provided
    test = test.reset_index().rename(columns={'index': 'sample_id'})
    submission = test[['sample_id', 'price']]

# Save predictions for submission
submission.to_csv('test_out.csv', index=False)
print(" test_out.csv file saved successfully with sample_id and predicted_price!")

