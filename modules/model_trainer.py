import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

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
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.model = LinearRegression()
            self._model_loaded = False
            self._initialized = True

    def load_model(self, csv_path):
        """Initialize and train the model at startup"""
        try:
            logger.info("Loading and training model...")
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Essay', 'Overall'])

            X_train, _, y_train, _ = train_test_split(
                df['Essay'], 
                df['Overall'],
                test_size=0.2,
                random_state=42
            )

            # Train the model
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_tfidf, y_train)
            self._model_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict_score(self, essay_text):
        """Thread-safe prediction with error handling"""
        if not self._model_loaded:
            logger.error("Model not loaded. Call load_model() first")
            return None

        try:
            with self._lock:
                text_tfidf = self.vectorizer.transform([essay_text])
                score = self.model.predict(text_tfidf)[0]
                return round(float(score), 1)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

# Initialize model at module level
model_trainer = ModelTrainer()