
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class SimpleEssayScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LinearRegression()
        self._model_loaded = False

    def train_model(self, csv_path):
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

            # Vectorize text
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)

            # Train model
            self.model.fit(X_train_tfidf, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train_tfidf)
            test_pred = self.model.predict(X_test_tfidf)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            logger.info(f"Train RMSE: {train_rmse:.2f}")
            logger.info(f"Test RMSE: {test_rmse:.2f}")
            
            self._model_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            return False

    def predict_score(self, essay_text):
        if not self._model_loaded:
            logger.error("Model not trained. Call train_model() first")
            return None

        try:
            # Vectorize input text
            text_tfidf = self.vectorizer.transform([essay_text])
            # Predict score
            score = self.model.predict(text_tfidf)[0]
            return round(float(score), 1)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

# Initialize model
simple_model = SimpleEssayScorer()
