
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LinearRegression()
        self._lock = threading.Lock()
        
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts using TF-IDF"""
        with self._lock:
            return self.vectorizer.transform(texts)
    
    def train_model(self, csv_path, max_workers=4):
        try:
            # Load and preprocess data
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Essay', 'Overall'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['Essay'], 
                df['Overall'],
                test_size=0.2,
                random_state=42
            )
            
            # Fit vectorizer on training data
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Train model
            self.model.fit(X_train_tfidf, y_train)
            
            # Parallel prediction
            def predict_batch(X_batch):
                return self.model.predict(X_batch)
            
            # Split test data into batches
            batch_size = len(X_test_tfidf) // max_workers
            X_batches = [X_test_tfidf[i:i + batch_size] for i in range(0, len(X_test_tfidf), batch_size)]
            
            # Parallel prediction using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                predictions = list(executor.map(predict_batch, X_batches))
            
            # Combine predictions
            y_pred = np.concatenate(predictions)
            
            # Calculate metrics
            mse = mean_squared_error(y_test[:len(y_pred)], y_pred)
            mae = mean_absolute_error(y_test[:len(y_pred)], y_pred)
            
            return {
                'mse': mse,
                'mae': mae,
                'model': self.model,
                'vectorizer': self.vectorizer
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return None
    
    def predict_score(self, essay_text):
        """Predict score for a single essay"""
        try:
            # Vectorize the text
            text_tfidf = self.vectorizer.transform([essay_text])
            # Predict
            score = self.model.predict(text_tfidf)[0]
            return round(score, 1)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None
