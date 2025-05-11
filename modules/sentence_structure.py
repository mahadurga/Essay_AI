import logging
import nltk
import textstat
import spacy
import re
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    try:
        # Fall back to loading a different model if needed
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    except Exception as e2:
        logger.error(f"Error loading fallback spaCy model: {str(e2)}")
        nlp = None

def analyze_sentence_structure(text):
    """
    Analyze sentence structure, complexity, and readability
    
    Args:
        text (str): The essay text to analyze
        
    Returns:
        dict: Dictionary containing sentence structure analysis
    """
    try:
        if not text:
            return {
                "error": "No text provided",
                "score": 0,
                "metrics": {},
                "suggestions": []
            }
            
        # Tokenize text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Calculate sentence length statistics
        sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
        avg_sentence_length = sum(sentence_lengths) / max(1, len(sentence_lengths))
        
        # Calculate readability scores
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        
        # Calculate sentence type distribution if spaCy is available
        sentence_complexity = {}
        passive_count = 0
        transition_words_count = 0
        
        if nlp:
            # Process with spaCy
            doc = nlp(text)
            
            # Count passive voice
            passive_count = sum(1 for token in doc if token.dep_ == "nsubjpass")
            
            # Count transition words
            transition_words = [
                "however", "therefore", "furthermore", "moreover", "consequently",
                "nevertheless", "in addition", "similarly", "conversely", "meanwhile",
                "subsequently", "meanwhile", "accordingly", "thus", "hence",
                "in contrast", "for example", "in conclusion"
            ]
            transition_words_count = sum(1 for token in doc if token.text.lower() in transition_words)
            
            # Analyze sentence types
            simple_count = 0
            compound_count = 0
            complex_count = 0
            
            for sent in doc.sents:
                # Count clauses based on verb presence
                clause_markers = [token for token in sent if token.pos_ == "VERB"]
                
                if len(clause_markers) == 1:
                    simple_count += 1
                elif len(clause_markers) == 2:
                    compound_count += 1
                else:
                    complex_count += 1
                    
            sentence_complexity = {
                "simple": simple_count / max(1, len(list(doc.sents))),
                "compound": compound_count / max(1, len(list(doc.sents))),
                "complex": complex_count / max(1, len(list(doc.sents)))
            }
        
        # Calculate sentence variety score (0-10)
        length_variety = min(10, (max(1, len(set(sentence_lengths))) / max(1, len(sentence_lengths))) * 10)
        
        # Calculate overall structure score
        readability_score = min(10, max(0, (flesch_reading_ease / 10)))
        
        # Detect sentence starters
        sentence_starters = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            if words:
                sentence_starters.append(words[0].lower())
        
        starter_variety = min(10, (len(set(sentence_starters)) / max(1, len(sentence_starters))) * 10)
        
        # Overall score is a weighted average of different metrics
        structure_score = (
            readability_score * 0.4 +
            length_variety * 0.3 +
            starter_variety * 0.3
        )
        
        # Generate suggestions
        suggestions = []
        
        if avg_sentence_length > 30:
            suggestions.append("Consider breaking down long sentences for better readability")
        elif avg_sentence_length < 10:
            suggestions.append("Try using more varied sentence lengths, including some longer sentences")
            
        if flesch_reading_ease < 50:
            suggestions.append("Your writing may be too complex; consider simplifying your language")
            
        if nlp and sentence_complexity.get("complex", 0) > 0.7:
            suggestions.append("Your essay has many complex sentences; consider mixing in simpler structures")
            
        if len(set(sentence_starters)) < len(sentence_starters) * 0.5:
            suggestions.append("Try varying your sentence starters for more engaging writing")
            
        if nlp and passive_count > len(list(doc.sents)) * 0.3:
            suggestions.append("Consider using more active voice in your writing")
            
        if nlp and transition_words_count < len(list(doc.sents)) * 0.15:
            suggestions.append("Consider using more transition words to improve flow between sentences")
        
        return {
            "score": structure_score,
            "metrics": {
                "avg_sentence_length": avg_sentence_length,
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "gunning_fog": gunning_fog,
                "sentence_complexity": sentence_complexity,
                "passive_voice_count": passive_count,
                "transition_words_count": transition_words_count,
                "sentence_starter_variety": starter_variety
            },
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Error in sentence structure analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "score": 0,
            "metrics": {},
            "suggestions": []
        }
