import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import DistilBertTokenizer, DistilBertModel
import logging
import spacy
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EssayModelTrainer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def get_bert_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding[0])
        return np.array(embeddings)

    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Essay', 'Overall'])

        X = [self.preprocess_text(text) for text in df['Essay']]
        y = df['Overall'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Generating BERT embeddings...")
        X_train_emb = self.get_bert_embeddings(X_train)
        X_test_emb = self.get_bert_embeddings(X_test)

        X_train_scaled = self.scaler.fit_transform(X_train_emb)
        X_test_scaled = self.scaler.transform(X_test_emb)

        logger.info("Training regressor...")
        self.regressor.fit(X_train_scaled, y_train)

        y_pred = self.regressor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Test MSE: {mse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")

        return mse, mae

    def predict(self, essay_text):
        processed_text = self.preprocess_text(essay_text)
        embedding = self.get_bert_embeddings([processed_text])
        embedding_scaled = self.scaler.transform(embedding)
        score = self.regressor.predict(embedding_scaled)[0]
        return float(score)

if __name__ == "__main__":
    trainer = EssayModelTrainer()
    trainer.train("ielts_writing_dataset.csv")