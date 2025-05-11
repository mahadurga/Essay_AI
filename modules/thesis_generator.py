import logging
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
import spacy

# Configure logger
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If model not found, use a simpler approach
    logger.warning("SpaCy model not found. Using default English model.")
    nlp = spacy.blank('en')  # Use blank model as fallback

def extract_keywords(text, n=5):
    """
    Extract the most important keywords from text
    
    Args:
        text (str): Input text
        n (int): Number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    # Tokenize and lowercase
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'it', 'for'])
        
    # Process with spaCy to get lemmas and filter out non-content words
    doc = nlp(text.lower())
    
    # Get content words (nouns, verbs, adjectives, adverbs)
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    
    # Count lemmas
    lemma_counts = Counter()
    for token in doc:
        if (token.pos_ in content_pos and 
            not token.is_stop and 
            token.lemma_.isalpha() and
            len(token.lemma_) > 1):
            lemma_counts[token.lemma_] += 1
    
    # Return most common keywords
    return [word for word, count in lemma_counts.most_common(n)]

def identify_main_argument(text):
    """
    Identify the main argument or focus of an essay
    
    Args:
        text (str): The essay text
        
    Returns:
        dict: Dictionary with main argument and supporting details
    """
    # Get sentences and extract potential thesis statements
    sentences = sent_tokenize(text)
    
    # Simple heuristic - look for sentences in the first and last paragraphs
    # as these often contain the thesis or main point
    first_para = ' '.join(sentences[:min(3, len(sentences))])
    last_para = ' '.join(sentences[-min(3, len(sentences)):])
    
    # Get keywords from the whole text to identify main topics
    keywords = extract_keywords(text, 7)
    
    # Process first and last paragraphs with spaCy to extract key entities and concepts
    first_doc = nlp(first_para)
    last_doc = nlp(last_para)
    
    # Extract noun chunks as potential topics
    topics = []
    for chunk in first_doc.noun_chunks:
        if len(chunk.text.split()) > 1:  # Get multi-word phrases
            topics.append(chunk.text)
    
    if len(topics) < 3:  # If not enough from first paragraph, add from last
        for chunk in last_doc.noun_chunks:
            if len(chunk.text.split()) > 1 and chunk.text not in topics:
                topics.append(chunk.text)
                if len(topics) >= 3:
                    break
    
    # Default values
    main_topic = "the essay topic" if not topics else topics[0]
    thesis = ""
    
    # Look for sentences with typical thesis patterns
    thesis_indicators = [
        "argue", "claim", "suggest", "believe", "think", 
        "main", "focus", "purpose", "aim", "goal", "thesis",
        "important", "significant", "crucial", "essential"
    ]
    
    for sentence in sentences[:min(5, len(sentences))]:
        if any(indicator in sentence.lower() for indicator in thesis_indicators):
            thesis = sentence
            break
    
    if not thesis and len(sentences) > 0:
        # If no clear thesis found, take first or second sentence
        thesis = sentences[min(1, len(sentences)-1)]
    
    return {
        "main_topic": main_topic,
        "keywords": keywords,
        "potential_topics": topics[:3],
        "thesis_statement": thesis
    }

def generate_thesis(text, style="academic"):
    """
    Generate a thesis statement based on essay content
    
    Args:
        text (str): The essay text
        style (str): Style of thesis ("academic", "concise", or "descriptive")
        
    Returns:
        str: Generated thesis statement
    """
    # Extract key information from the essay
    info = identify_main_argument(text)
    
    # Build thesis statement based on style
    thesis = ""
    
    if style == "academic":
        # Academic style with formal language
        keywords = ", ".join(info["keywords"][:3])
        topics = " and ".join(info["potential_topics"][:2])
        
        if topics:
            thesis = f"This essay examines {topics}, analyzing the relationship between {keywords} to argue that {info['thesis_statement']}"
        else:
            thesis = f"This essay analyzes {keywords} to demonstrate that {info['thesis_statement']}"
            
    elif style == "concise":
        # Short and direct thesis
        keywords = " and ".join(info["keywords"][:2])
        thesis = f"This essay argues that {keywords} are central to understanding {info['main_topic']}."
        
    else:  # descriptive
        # Descriptive thesis that outlines the essay structure
        keywords = ", ".join(info["keywords"][:4])
        thesis = f"This essay explores {info['main_topic']}, focusing on {keywords}, and concludes by suggesting {info['thesis_statement']}"
    
    return thesis

def generate_summary(text, length="medium"):
    """
    Generate a summary of the essay
    
    Args:
        text (str): The essay text
        length (str): Length of summary ("short", "medium", or "long")
        
    Returns:
        str: Generated summary
    """
    # Extract main arguments and information
    info = identify_main_argument(text)
    
    # Extract sentences for summary using a simple extractive approach
    sentences = sent_tokenize(text)
    
    # Process the entire text
    doc = nlp(text)
    
    # Score sentences based on keyword frequency and position
    sent_scores = {}
    for i, sent in enumerate(sentences):
        sent_doc = nlp(sent)
        score = 0
        
        # Position score - introduction and conclusion sentences get higher scores
        if i < 3:
            score += 0.3  # Introduction sentences
        elif i > len(sentences) - 4:
            score += 0.3  # Conclusion sentences
            
        # Keyword score
        for keyword in info["keywords"]:
            if keyword in sent.lower():
                score += 0.2
                
        # Length score - prefer medium length sentences
        if 10 <= len(sent.split()) <= 25:
            score += 0.1
            
        sent_scores[sent] = score
    
    # Determine number of sentences to include based on length parameter
    if length == "short":
        n_sentences = min(2, len(sentences))
    elif length == "medium":
        n_sentences = min(3, len(sentences))
    else:  # long
        n_sentences = min(5, len(sentences))
    
    # Get top sentences while keeping original order
    top_sentences = [s for s, _ in sorted(sent_scores.items(), key=lambda x: -x[1])[:n_sentences*2]]
    selected_sentences = [s for s in sentences if s in top_sentences][:n_sentences]
    
    # Order by position in text
    selected_sentences.sort(key=lambda x: sentences.index(x))
    
    # Join sentences into a summary
    summary = " ".join(selected_sentences)
    
    return summary