import logging
import nltk
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

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

def analyze_coherence(text):
    """
    Analyze the coherence and organization of an essay
    
    Args:
        text (str): The essay text to analyze
        
    Returns:
        dict: Dictionary containing coherence analysis
    """
    try:
        if not text:
            return {
                "error": "No text provided",
                "score": 0,
                "metrics": {},
                "suggestions": []
            }
            
        # Tokenize text into sentences and paragraphs
        sentences = nltk.sent_tokenize(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If text doesn't have clear paragraph breaks, estimate them
        if len(paragraphs) <= 1:
            # Estimate paragraphs (roughly every 3-5 sentences)
            paragraph_size = max(1, min(5, len(sentences) // 3))
            paragraphs = [' '.join(sentences[i:i+paragraph_size]) 
                         for i in range(0, len(sentences), paragraph_size)]
        
        # Calculate metrics
        metrics = {}
        
        # 1. Paragraph structure analysis
        metrics["paragraph_count"] = len(paragraphs)
        metrics["sentences_per_paragraph"] = [len(nltk.sent_tokenize(p)) for p in paragraphs]
        metrics["avg_sentences_per_paragraph"] = sum(metrics["sentences_per_paragraph"]) / max(1, len(metrics["sentences_per_paragraph"]))
        
        # 2. Topic consistency within paragraphs
        paragraph_consistency = []
        
        # If we have spaCy available, use more sophisticated methods
        if nlp:
            # Get paragraph embeddings and measure consistency
            paragraph_embeddings = []
            for paragraph in paragraphs:
                doc = nlp(paragraph)
                # Get average of word vectors
                if doc.vector.any():
                    paragraph_embeddings.append(doc.vector)
                else:
                    # Fallback if no vectors available
                    paragraph_embeddings.append(np.zeros(nlp.vocab.vectors.shape[1]))
            
            # Calculate coherence between adjacent paragraphs
            coherence_scores = []
            for i in range(len(paragraph_embeddings)-1):
                v1 = paragraph_embeddings[i]
                v2 = paragraph_embeddings[i+1]
                
                # Calculate cosine similarity if vectors are not zero
                if np.sum(v1) == 0 or np.sum(v2) == 0:
                    coherence_scores.append(0.5)  # Default value
                else:
                    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    coherence_scores.append(max(0, min(1, cosine_sim)))
            
            metrics["paragraph_coherence_scores"] = coherence_scores
            metrics["avg_paragraph_coherence"] = sum(coherence_scores) / max(1, len(coherence_scores))
        
        # 3. Transition words analysis
        transition_words = [
            "however", "therefore", "furthermore", "moreover", "consequently",
            "nevertheless", "in addition", "similarly", "conversely", "meanwhile",
            "subsequently", "accordingly", "thus", "hence", "in contrast", 
            "for example", "in conclusion", "in summary", "finally", "first",
            "second", "third", "next", "then", "lastly", "to summarize"
        ]
        
        transition_count = 0
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            for word in transition_words:
                if word in ' '.join(words):
                    transition_count += 1
                    break
        
        metrics["transition_word_ratio"] = transition_count / max(1, len(sentences))
        
        # 4. Essay structure analysis
        has_introduction = False
        has_conclusion = False
        
        # Check for introduction and conclusion based on typical patterns
        intro_words = ["introduction", "introduce", "begin", "start", "first"]
        conclusion_words = ["conclusion", "conclude", "summary", "summarize", "finally", "in summary", "to conclude"]
        
        # Simple heuristic: check first and last paragraphs
        if paragraphs:
            first_para = paragraphs[0].lower()
            has_introduction = any(word in first_para for word in intro_words) or len(paragraphs) >= 3
            
            last_para = paragraphs[-1].lower()
            has_conclusion = any(word in last_para for word in conclusion_words) or len(paragraphs) >= 3
        
        metrics["has_introduction"] = has_introduction
        metrics["has_conclusion"] = has_conclusion
        
        # 5. Calculate overall coherence score (0-10)
        coherence_score = 0
        components = []
        
        # Add paragraph coherence if available
        if "avg_paragraph_coherence" in metrics:
            components.append(metrics["avg_paragraph_coherence"] * 10)
        
        # Add transition word score
        components.append(min(10, metrics["transition_word_ratio"] * 20))
        
        # Add structure score
        structure_score = 5
        if metrics["has_introduction"]:
            structure_score += 2.5
        if metrics["has_conclusion"]:
            structure_score += 2.5
        components.append(structure_score)
        
        # Calculate average score
        coherence_score = sum(components) / len(components)
        
        # Generate suggestions
        suggestions = []
        
        if not metrics["has_introduction"]:
            suggestions.append("Consider adding a clear introduction that sets up your main argument")
            
        if not metrics["has_conclusion"]:
            suggestions.append("Your essay would benefit from a conclusion that summarizes your key points")
            
        if metrics["transition_word_ratio"] < 0.2:
            suggestions.append("Try using more transition words to improve the flow between ideas")
            
        if "avg_paragraph_coherence" in metrics and metrics["avg_paragraph_coherence"] < 0.5:
            suggestions.append("Work on making your paragraphs flow more smoothly from one to the next")
            
        if metrics["avg_sentences_per_paragraph"] > 7:
            suggestions.append("Consider breaking up longer paragraphs for better readability")
        elif metrics["avg_sentences_per_paragraph"] < 2:
            suggestions.append("Try developing your paragraphs more fully with supporting details")
            
        return {
            "score": coherence_score,
            "metrics": metrics,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Error in coherence analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "score": 0,
            "metrics": {},
            "suggestions": []
        }
