import os
import logging
import concurrent.futures
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from modules.grammar_checker import check_grammar_spelling
from modules.sentence_structure import analyze_sentence_structure
from modules.coherence_analyzer import analyze_coherence
from modules.vocabulary_analyzer import analyze_vocabulary
from modules.essay_scorer import score_essay
from modules.feedback_generator import generate_feedback
from modules.dataset_loader import get_sample_essays_from_dataset
from modules.model_trainer import model_trainer

# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_for_testing")

# Thread pool executor for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_essay():
    """
    Analyze the submitted essay using multithreading for parallel processing
    Returns JSON with detailed feedback
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        essay_text = data.get('essay', '')
        if not essay_text:
            return jsonify({'error': 'No essay text provided'}), 400
            
        logger.info(f"Starting analysis for essay of length: {len(essay_text)}")

        logger.debug(f"Received essay with length: {len(essay_text)} characters")

        # Create a thread for each analysis component
        grammar_future = executor.submit(check_grammar_spelling, essay_text)
        structure_future = executor.submit(analyze_sentence_structure, essay_text)
        coherence_future = executor.submit(analyze_coherence, essay_text)
        vocabulary_future = executor.submit(analyze_vocabulary, essay_text)

        # Get results from all threads
        grammar_results = grammar_future.result()
        structure_results = structure_future.result()
        coherence_results = coherence_future.result()
        vocabulary_results = vocabulary_future.result()

        # Calculate overall score based on component results
        score_results = score_essay(
            grammar_results,
            structure_results,
            coherence_results,
            vocabulary_results
        )

        # Generate comprehensive feedback
        feedback = generate_feedback(
            essay_text,
            grammar_results,
            structure_results,
            coherence_results,
            vocabulary_results,
            score_results
        )

        # Convert numpy types to native Python types for JSON serialization
        feedback = convert_numpy_types(feedback)

        return jsonify(feedback)

    except Exception as e:
        logger.error(f"Error analyzing essay: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while analyzing the essay',
            'message': str(e)
        }), 500

@app.route('/generate_thesis', methods=['POST'])
def generate_thesis_endpoint():
    """
    Generate a thesis statement or summary based on the submitted essay

    Returns:
        JSON with generated thesis statement and summary
    """
    try:
        data = request.get_json()
        essay_text = data.get('essay', '')
        style = data.get('style', 'academic')  # academic, concise, descriptive
        summary_length = data.get('summary_length', 'medium')  # short, medium, long

        if not essay_text:
            return jsonify({'error': 'No essay text provided'}), 400

        logger.debug(f"Generating thesis for essay with length: {len(essay_text)} characters")

        # Import here to avoid circular imports
        from modules.thesis_generator import generate_thesis, generate_summary, identify_main_argument

        # Get main argument information
        argument_info = identify_main_argument(essay_text)

        # Generate thesis and summary
        thesis = generate_thesis(essay_text, style)
        summary = generate_summary(essay_text, summary_length)

        # Prepare response
        response = {
            'thesis_statement': thesis,
            'summary': summary,
            'main_topic': argument_info['main_topic'],
            'keywords': argument_info['keywords'],
            'potential_topics': argument_info['potential_topics']
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating thesis: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while generating thesis',
            'message': str(e)
        }), 500

@app.route('/sample_essays', methods=['GET'])
def get_sample_essays():
    """Return a list of sample essays from the IELTS dataset"""
    try:
        # Get samples from the IELTS dataset
        samples = get_sample_essays_from_dataset(5)
        return jsonify(samples)
    except Exception as e:
        logger.error(f"Error fetching sample essays: {str(e)}")
        # Fallback samples if dataset loading fails
        samples = [
            {
                "title": "The Impact of Technology on Education",
                "text": "Technology has revolutionized the educational landscape in numerous ways. Digital classrooms, online resources, and interactive learning tools have transformed how students engage with content. However, this digital transformation also presents challenges such as digital divides and concerns about screen time. Despite these concerns, the benefits of educational technology—including personalized learning, global connectivity, and enhanced engagement—suggest that technological integration in education will continue to evolve. The key challenge for educators is to leverage technology effectively while maintaining the human elements of teaching and learning that are essential to education."
            },
            {
                "title": "Climate Change: A Global Crisis",
                "text": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, extreme weather events, and melting ice caps provide undeniable evidence of this phenomenon. Human activities, particularly the burning of fossil fuels and deforestation, have significantly contributed to greenhouse gas emissions, exacerbating the natural warming cycle. The consequences of unchecked climate change are severe, potentially leading to habitat destruction, food insecurity, and rising sea levels. Addressing this crisis requires coordinated international efforts to reduce carbon emissions, transition to renewable energy sources, and implement sustainable practices across all sectors of society."
            }
        ]
        return jsonify(samples)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)