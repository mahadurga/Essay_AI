import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class SimpleModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train_model(self, csv_file):
        """
        Trains a linear regression model on the essay data.

        Args:
            csv_file (str): Path to the CSV file containing essay data.
        """
        try:
            # Load the dataset
            df = pd.read_csv(csv_file)

            # Handle missing values
            df = df.dropna(subset=['EssayText', 'Overall'])

            # Separate features (essays) and target (overall score)
            essays = df['EssayText']
            overall_scores = df['Overall']

            # Split data into training and testing sets
            essays_train, essays_test, scores_train, scores_test = train_test_split(
                essays, overall_scores, test_size=0.2, random_state=42
            )

            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

            # Fit and transform the training essays
            essays_train_tfidf = self.vectorizer.fit_transform(essays_train)

            # Transform the testing essays
            essays_test_tfidf = self.vectorizer.transform(essays_test)

            # Initialize and train the linear regression model
            self.model = LinearRegression()
            self.model.fit(essays_train_tfidf, scores_train)

            # Evaluate the model
            self.evaluate_model(essays_test_tfidf, scores_test)

        except Exception as e:
            print(f"Error training the model: {e}")

    def evaluate_model(self, essays_test_tfidf, scores_test):
        """
        Evaluates the trained model on the test set.

        Args:
            essays_test_tfidf: TF-IDF vectorized test essays.
            scores_test: Actual overall scores for the test essays.
        """
        try:
            # Make predictions on the test set
            scores_pred = self.model.predict(essays_test_tfidf)

            # Calculate evaluation metrics
            mse = mean_squared_error(scores_test, scores_pred)
            r2 = r2_score(scores_test, scores_pred)

            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")

        except Exception as e:
            print(f"Error evaluating the model: {e}")

    def predict_score(self, essay_text):
        """
        Predicts the overall score for a given essay text.

        Args:
            essay_text (str): The essay text to predict the score for.

        Returns:
            float: The predicted overall score.
        """
        try:
            # Transform the essay text using the trained vectorizer
            essay_tfidf = self.vectorizer.transform([essay_text])

            # Predict the score using the trained model
            predicted_score = self.model.predict(essay_tfidf)[0]

            return predicted_score

        except Exception as e:
            print(f"Error predicting the score: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    # Create an instance of the SimpleModel
    simple_model = SimpleModel()

    # Train the model using the essay data
    simple_model.train_model('ielts_writing_dataset.csv')

    # Example essay for prediction
    example_essay = "Technology has greatly improved our lives. It has made communication easier and faster, and it has also provided us with access to a wealth of information. However, technology has also had some negative effects. It has made us more isolated and less active, and it has also led to new forms of crime."

    # Predict the score for the example essay
    predicted_score = simple_model.predict_score(example_essay)

    if predicted_score is not None:
        print(f"Predicted Overall Score: {predicted_score}")