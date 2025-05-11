import logging
import nltk
import string
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")
    stop_words = set()
    lemmatizer = None

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    try:
        # Fall back to loading a different model
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    except Exception as e2:
        logger.error(f"Error loading fallback spaCy model: {str(e2)}")
        nlp = None

def analyze_vocabulary(text):
    """
    Analyze vocabulary richness, diversity, and appropriate usage
    
    Args:
        text (str): The essay text to analyze
        
    Returns:
        dict: Dictionary containing vocabulary analysis
    """
    try:
        if not text:
            return {
                "error": "No text provided",
                "score": 0,
                "metrics": {},
                "suggestions": []
            }
            
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        tokens = [word for word in tokens if word not in string.punctuation]
        content_tokens = [word for word in tokens if word not in stop_words]
        
        # Calculate basic lexical diversity metrics
        total_words = len(tokens)
        unique_words = len(set(tokens))
        content_words = len(content_tokens)
        unique_content_words = len(set(content_tokens))
        
        # Calculate Type-Token Ratio (TTR)
        ttr = unique_words / max(1, total_words)
        content_ttr = unique_content_words / max(1, content_words)
        
        # Get word frequency
        word_freq = Counter(content_tokens)
        most_common = word_freq.most_common(10)
        
        # Advanced vocabulary analysis with spaCy if available
        advanced_metrics = {}
        if nlp:
            doc = nlp(text)
            
            # Part of speech distribution
            pos_counts = Counter([token.pos_ for token in doc])
            
            # Calculate adjective and adverb ratios
            adj_ratio = pos_counts.get('ADJ', 0) / max(1, len(doc))
            adv_ratio = pos_counts.get('ADV', 0) / max(1, len(doc))
            
            # Extract advanced metrics
            advanced_metrics = {
                "pos_distribution": dict(pos_counts),
                "adjective_ratio": adj_ratio,
                "adverb_ratio": adv_ratio
            }
            
            # Find repeated words in close proximity
            repeated_proximity = []
            window_size = 50  # characters
            
            for i, token in enumerate(doc):
                if token.is_alpha and len(token.text) > 3 and token.pos_ not in ['DET', 'ADP', 'CCONJ', 'PART']:
                    window_text = text[max(0, token.idx - window_size):min(len(text), token.idx + window_size)]
                    occurrences = window_text.lower().count(token.text.lower())
                    if occurrences > 2:
                        repeated_proximity.append(token.text)
                        
            advanced_metrics["repeated_proximity"] = list(set(repeated_proximity))
        
        # Calculate vocabulary score (0-10)
        # Base on lexical diversity and presence of advanced vocabulary
        vocab_score = min(10, ttr * 15)  # TTR typically ranges from 0.4 to 0.7 for essays
        
        # Calculate sophistication based on word length
        avg_word_length = sum(len(word) for word in content_tokens) / max(1, len(content_tokens))
        length_score = min(10, avg_word_length)
        
        # Adjust score based on advanced metrics if available
        if advanced_metrics:
            # Reward appropriate use of adjectives and adverbs without excess
            adj_adv_score = 10 - (abs(advanced_metrics["adjective_ratio"] - 0.1) * 30 + 
                                  abs(advanced_metrics["adverb_ratio"] - 0.05) * 30)
            
            # Penalize for excessive repetition
            repetition_penalty = min(2, len(advanced_metrics.get("repeated_proximity", [])) * 0.5)
            
            vocab_score = (vocab_score * 0.6 + 
                           length_score * 0.2 + 
                           max(0, adj_adv_score) * 0.2 - 
                           repetition_penalty)
        
        # Generate suggestions
        suggestions = []
        
        if ttr < 0.4:
            suggestions.append("Try using more varied vocabulary instead of repeating the same words")
            
        if advanced_metrics and advanced_metrics["adverb_ratio"] > 0.1:
            suggestions.append("Consider reducing the number of adverbs and using stronger verbs instead")
            
        if advanced_metrics and len(advanced_metrics.get("repeated_proximity", [])) > 0:
            suggestions.append("Avoid repeating these words in close proximity: " + 
                               ", ".join(advanced_metrics["repeated_proximity"][:3]))
            
        if avg_word_length < 4.5:
            suggestions.append("Consider incorporating more sophisticated vocabulary where appropriate")
            
        # Check for common filler phrases
        filler_phrases = ["in my opinion", "i think that", "i believe that", "in conclusion", 
                         "as a matter of fact", "for all intents and purposes"]
        found_fillers = []
        
        for phrase in filler_phrases:
            if phrase in text.lower():
                found_fillers.append(phrase)
                
        if found_fillers:
            suggestions.append("Consider replacing these filler phrases with more precise language: " + 
                              ", ".join(found_fillers))
        
        return {
            "score": max(0, min(10, vocab_score)),
            "metrics": {
                "total_words": total_words,
                "unique_words": unique_words,
                "content_words": content_words,
                "unique_content_words": unique_content_words,
                "type_token_ratio": ttr,
                "content_type_token_ratio": content_ttr,
                "avg_word_length": avg_word_length,
                "most_common_words": most_common,
                **advanced_metrics
            },
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Error in vocabulary analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "score": 0,
            "metrics": {},
            "suggestions": []
        }
