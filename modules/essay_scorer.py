import logging
import math

# Configure logging
logger = logging.getLogger(__name__)

def score_essay(grammar_results, structure_results, coherence_results, vocabulary_results):
    """
    Calculate an overall essay score based on individual component scores
    
    Args:
        grammar_results (dict): Results from grammar analysis
        structure_results (dict): Results from sentence structure analysis
        coherence_results (dict): Results from coherence analysis
        vocabulary_results (dict): Results from vocabulary analysis
        
    Returns:
        dict: Dictionary containing the overall essay score and breakdown
    """
    try:
        # Extract scores from each component
        grammar_score = grammar_results.get("score", 0)
        structure_score = structure_results.get("score", 0)
        coherence_score = coherence_results.get("score", 0)
        vocabulary_score = vocabulary_results.get("score", 0)
        
        # Define weightings for each component
        weights = {
            "grammar": 0.25,
            "structure": 0.25,
            "coherence": 0.3,
            "vocabulary": 0.2
        }
        
        # Calculate weighted average score
        overall_score = (
            grammar_score * weights["grammar"] +
            structure_score * weights["structure"] +
            coherence_score * weights["coherence"] +
            vocabulary_score * weights["vocabulary"]
        )
        
        # Round to one decimal place
        overall_score = round(overall_score, 1)
        
        # Determine grade based on score (A, B, C, D, F)
        grade = "F"
        if overall_score >= 9:
            grade = "A+"
        elif overall_score >= 8.5:
            grade = "A"
        elif overall_score >= 8:
            grade = "A-"
        elif overall_score >= 7.5:
            grade = "B+"
        elif overall_score >= 7:
            grade = "B"
        elif overall_score >= 6.5:
            grade = "B-"
        elif overall_score >= 6:
            grade = "C+"
        elif overall_score >= 5.5:
            grade = "C"
        elif overall_score >= 5:
            grade = "C-"
        elif overall_score >= 4:
            grade = "D"
        
        # Determine strongest and weakest areas
        scores = {
            "grammar": grammar_score,
            "structure": structure_score,
            "coherence": coherence_score,
            "vocabulary": vocabulary_score
        }
        
        strongest_area = max(scores, key=scores.get)
        weakest_area = min(scores, key=scores.get)
        
        # Generate score breakdown
        score_breakdown = {
            "grammar": grammar_score,
            "structure": structure_score,
            "coherence": coherence_score,
            "vocabulary": vocabulary_score
        }
        
        return {
            "overall_score": overall_score,
            "grade": grade,
            "score_breakdown": score_breakdown,
            "strongest_area": strongest_area,
            "weakest_area": weakest_area
        }
        
    except Exception as e:
        logger.error(f"Error in essay scoring: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "overall_score": 0,
            "grade": "F",
            "score_breakdown": {
                "grammar": 0,
                "structure": 0,
                "coherence": 0,
                "vocabulary": 0
            },
            "strongest_area": None,
            "weakest_area": None
        }
