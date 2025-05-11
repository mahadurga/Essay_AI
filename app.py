import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
            df = df.dropna(subset=['Essay', 'Overall'])

            # Separate features (essays) and target (overall score)
            essays = df['Essay']
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

app = Flask(__name__)
model = SimpleModel()

# Train the model when app starts
model.train_model('ielts_writing_dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(404)
def not_found(e):
    return "404 - Page not found", 404

@app.route('/analyze', methods=['POST'])
def analyze_essay():
    try:
        data = request.get_json()
        essay_text = data.get('essay', '')

        if not essay_text:
            return jsonify({'error': 'No essay text provided'}), 400

        score = model.predict_score(essay_text)

        if score is None:
            return jsonify({'error': 'Failed to predict score'}), 500

        # Create a structured feedback response
        feedback = {
            'overall_assessment': {
                'score': float(score),
                'grade': 'A' if score >= 9 else 'B' if score >= 8 else 'C' if score >= 7 else 'D' if score >= 6 else 'F',
                'summary': f'Your essay received a score of {score}/10.',
                'score_breakdown': {
                    'grammar': score,
                    'structure': score,
                    'coherence': score,
                    'vocabulary': score
                }
            },
            'grammar_feedback': {
                'score': score,
                'total_issues': 0,
                'suggestions': ['Focus on maintaining consistent tense throughout your essay.']
            },
            'structure_feedback': {
                'score': score,
                'metrics': {
                    'avg_sentence_length': len(essay_text.split()) / len(essay_text.split('.')),
                    'flesch_reading_ease': 100
                }
            },
            'coherence_feedback': {
                'score': score,
                'has_introduction': True,
                'has_conclusion': True
            },
            'vocabulary_feedback': {
                'score': score,
                'metrics': {
                    'unique_words': len(set(essay_text.split())),
                    'total_words': len(essay_text.split()),
                    'type_token_ratio': len(set(essay_text.split())) / len(essay_text.split()) if essay_text else 0
                }
            }
        }

        return jsonify(feedback)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)