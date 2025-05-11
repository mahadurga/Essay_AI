import logging
import language_tool_python
from concurrent.futures import ThreadPoolExecutor

# Configure logger
logger = logging.getLogger(__name__)

# Initialize the language tool
try:
    language_tool = language_tool_python.LanguageTool('en-US')
except Exception as e:
    logger.error(f"Error initializing LanguageTool: {str(e)}")
    language_tool = None

def check_grammar_spelling(text):
    """
    Check grammar and spelling in the essay text
    
    Args:
        text (str): The essay text to analyze
        
    Returns:
        dict: Dictionary containing grammar and spelling feedback
    """
    try:
        if not language_tool:
            return {
                "error": "Grammar checking tool not available",
                "score": 0,
                "issues": [],
                "suggestions": []
            }
            
        # Get grammar and spelling matches
        matches = language_tool.check(text)
        
        # Process the matches to extract useful information
        issues = []
        for match in matches:
            issues.append({
                "offset": match.offset,
                "length": match.errorLength,
                "message": match.message,
                "rule_id": match.ruleId,
                "sentence": text[max(0, match.offset-30):min(len(text), match.offset+match.errorLength+30)],
                "replacements": match.replacements[:3] if match.replacements else []
            })
        
        # Calculate a grammar score based on the number of issues relative to text length
        # Normalize to a scale of 0-10
        if len(text.split()) > 0:
            error_density = len(matches) / (len(text.split()) / 100)
            grammar_score = max(0, min(10, 10 - (error_density * 2)))
        else:
            grammar_score = 0
            
        # Generate suggestions based on common error types
        suggestions = generate_grammar_suggestions(issues)
        
        result = {
            "score": grammar_score,
            "issues": issues,
            "suggestions": suggestions,
            "total_issues": len(issues)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in grammar checking: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "score": 0,
            "issues": [],
            "suggestions": []
        }
        
def generate_grammar_suggestions(issues):
    """
    Generate actionable suggestions based on grammar issues
    
    Args:
        issues (list): List of grammar issues
        
    Returns:
        list: List of suggestion strings
    """
    suggestions = []
    
    # Count issue types
    error_types = {}
    for issue in issues:
        rule_id = issue["rule_id"]
        error_types[rule_id] = error_types.get(rule_id, 0) + 1
    
    # Generate suggestions based on frequency of error types
    if any(rule_id.startswith("AGREEMENT") for rule_id in error_types):
        suggestions.append("Focus on subject-verb agreement in your sentences")
    
    if any(rule_id.startswith("PUNCTUATION") for rule_id in error_types):
        suggestions.append("Review your punctuation usage, especially commas and periods")
    
    if any(rule_id.startswith("SPELLING") for rule_id in error_types):
        suggestions.append("Check your spelling carefully before submitting")
    
    if any(rule_id.startswith("PASSIVE_VOICE") for rule_id in error_types):
        suggestions.append("Consider using active voice for more direct and clear writing")
    
    if any(rule_id.startswith("WORDINESS") for rule_id in error_types):
        suggestions.append("Try to express your ideas more concisely")
    
    # Add a general suggestion if there are many issues
    if len(issues) > 10:
        suggestions.append("Consider using a grammar checker while writing to catch common errors")
    
    return suggestions
