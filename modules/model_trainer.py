
import logging
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import re
import threading

logger = logging.getLogger(__name__)

class EssayDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['score'] = self.scores[idx]
        return item

    def __len__(self):
        return len(self.scores)

class EssayScorer(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.regressor(pooled_output)

class ModelTrainer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = None
            self._model_loaded = False
            self._initialized = True

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def load_model(self, csv_path):
        try:
            logger.info("Loading and training model...")
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Essay', 'Overall'])

            texts = [self.preprocess_text(text) for text in df['Essay']]
            scores = df['Overall'].values

            X_train, X_val, y_train, y_val = train_test_split(
                texts, scores, test_size=0.2, random_state=42
            )

            train_dataset = EssayDataset(X_train, y_train, self.tokenizer)
            val_dataset = EssayDataset(X_val, y_val, self.tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.model = EssayScorer(bert_model).to(self.device)
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            criterion = nn.MSELoss()

            for epoch in range(3):
                self.model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    scores = batch['score'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), scores)
                    
                    loss.backward()
                    optimizer.step()

            self._model_loaded = True
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict_score(self, essay_text):
        if not self._model_loaded:
            logger.error("Model not loaded. Call load_model() first")
            return None

        try:
            with self._lock:
                self.model.eval()
                text = self.preprocess_text(essay_text)
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
                
                with torch.no_grad():
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    score = self.model(input_ids, attention_mask)
                    
                return round(float(score.squeeze().cpu().numpy()), 1)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

model_trainer = ModelTrainer()
