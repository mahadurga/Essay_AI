import os
import logging
import concurrent.futures
from flask import Flask, render_template, request, jsonify
from modules.grammar_checker import check_grammar_spelling
from modules.sentence_structure import analyze_sentence_structure
from modules.coherence_analyzer import analyze_coherence
from modules.vocabulary_analyzer import analyze_vocabulary
from modules.essay_scorer import score_essay
from modules.feedback_generator import generate_feedback

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
        essay_text = data.get('essay', '')
        
        if not essay_text:
            return jsonify({'error': 'No essay text provided'}), 400
        
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
        
        return jsonify(feedback)
        
    except Exception as e:
        logger.error(f"Error analyzing essay: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while analyzing the essay',
            'message': str(e)
        }), 500

@app.route('/sample_essays', methods=['GET'])
def get_sample_essays():
    """Return a list of sample essays for demonstration"""
    samples = [
        {
            "title": "The Impact of Technology on Education",
            "text": "Technology has revolutionized the educational landscape in numerous ways. Digital classrooms, online resources, and interactive learning tools have transformed how students engage with content. However, this digital transformation also presents challenges such as digital divides and concerns about screen time. Despite these concerns, the benefits of educational technology—including personalized learning, global connectivity, and enhanced engagement—suggest that technological integration in education will continue to evolve. The key challenge for educators is to leverage technology effectively while maintaining the human elements of teaching and learning that are essential to education."
        },
        {
            "title": "Climate Change: A Global Crisis",
            "text": "Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, extreme weather events, and melting ice caps provide undeniable evidence of this phenomenon. Human activities, particularly the burning of fossil fuels and deforestation, have significantly contributed to greenhouse gas emissions, exacerbating the natural warming cycle. The consequences of unchecked climate change are severe, potentially leading to habitat destruction, food insecurity, and rising sea levels. Addressing this crisis requires coordinated international efforts to reduce carbon emissions, transition to renewable energy sources, and implement sustainable practices across all sectors of society."
        },
        {
            "title": "The Role of Art in Society",
            "text": "Art has always played a vital role in human societies throughout history. As a form of expression, it transcends cultural and linguistic barriers, allowing for the communication of complex ideas and emotions. Art serves as a mirror to society, reflecting its values, concerns, and aspirations while simultaneously challenging conventions and provoking thought. Beyond its cultural significance, art contributes to individual well-being by fostering creativity, critical thinking, and emotional intelligence. In contemporary society, despite the prevalence of technology and practical concerns, art remains essential as a means of preserving cultural heritage, stimulating innovation, and enriching the human experience."
        }
    ]
    return jsonify(samples)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
