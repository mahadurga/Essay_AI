import logging
import random

# Configure logging
logger = logging.getLogger(__name__)

def generate_feedback(essay_text, grammar_results, structure_results, coherence_results, vocabulary_results, score_results):
    """
    Generate comprehensive feedback based on all analysis results
    
    Args:
        essay_text (str): The original essay text
        grammar_results (dict): Results from grammar analysis
        structure_results (dict): Results from sentence structure analysis
        coherence_results (dict): Results from coherence analysis
        vocabulary_results (dict): Results from vocabulary analysis
        score_results (dict): Results from scoring
        
    Returns:
        dict: Comprehensive feedback for the essay
    """
    try:
        # Create the comprehensive feedback structure
        feedback = {
            "overall_assessment": {
                "score": score_results.get("overall_score", 0),
                "grade": score_results.get("grade", "F"),
                "score_breakdown": score_results.get("score_breakdown", {}),
                "strongest_area": score_results.get("strongest_area"),
                "weakest_area": score_results.get("weakest_area")
            },
            "grammar_feedback": {
                "score": grammar_results.get("score", 0),
                "issues": grammar_results.get("issues", [])[:5],  # Limit to top 5 issues
                "suggestions": grammar_results.get("suggestions", []),
                "total_issues": grammar_results.get("total_issues", 0)
            },
            "structure_feedback": {
                "score": structure_results.get("score", 0),
                "metrics": {
                    "avg_sentence_length": structure_results.get("metrics", {}).get("avg_sentence_length", 0),
                    "flesch_reading_ease": structure_results.get("metrics", {}).get("flesch_reading_ease", 0),
                },
                "suggestions": structure_results.get("suggestions", [])
            },
            "coherence_feedback": {
                "score": coherence_results.get("score", 0),
                "has_introduction": coherence_results.get("metrics", {}).get("has_introduction", False),
                "has_conclusion": coherence_results.get("metrics", {}).get("has_conclusion", False),
                "suggestions": coherence_results.get("suggestions", [])
            },
            "vocabulary_feedback": {
                "score": vocabulary_results.get("score", 0),
                "metrics": {
                    "unique_words": vocabulary_results.get("metrics", {}).get("unique_words", 0),
                    "total_words": vocabulary_results.get("metrics", {}).get("total_words", 0),
                    "type_token_ratio": vocabulary_results.get("metrics", {}).get("type_token_ratio", 0),
                },
                "most_common_words": vocabulary_results.get("metrics", {}).get("most_common_words", []),
                "suggestions": vocabulary_results.get("suggestions", [])
            }
        }
        
        # Generate a summary paragraph based on the overall assessment
        summary = generate_summary_paragraph(score_results)
        feedback["overall_assessment"]["summary"] = summary
        
        # Generate priority improvements
        priority_improvements = generate_priority_improvements(
            grammar_results,
            structure_results,
            coherence_results,
            vocabulary_results,
            score_results
        )
        feedback["priority_improvements"] = priority_improvements
        
        # Add highlighted text with issues if available
        if grammar_results.get("issues"):
            feedback["highlighted_text"] = generate_highlighted_text(essay_text, grammar_results.get("issues", []))
        
        return feedback
        
    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "overall_assessment": {
                "score": 0,
                "grade": "F",
                "summary": "An error occurred while generating feedback."
            }
        }

def generate_summary_paragraph(score_results):
    """
    Generate a summary paragraph based on the overall score and breakdown
    
    Args:
        score_results (dict): Results from scoring
        
    Returns:
        str: A summary paragraph
    """
    overall_score = score_results.get("overall_score", 0)
    grade = score_results.get("grade", "F")
    strongest_area = score_results.get("strongest_area", "")
    weakest_area = score_results.get("weakest_area", "")
    
    # Determine the overall quality description
    quality = "needs significant improvement"
    if overall_score >= 9:
        quality = "excellent"
    elif overall_score >= 8:
        quality = "very good"
    elif overall_score >= 7:
        quality = "good"
    elif overall_score >= 6:
        quality = "satisfactory"
    elif overall_score >= 5:
        quality = "average"
    elif overall_score >= 4:
        quality = "below average"
    
    # Create a summary paragraph
    summary = f"Overall, your essay is {quality} with a score of {overall_score}/10 (grade {grade}). "
    
    # Add a comment about strongest area
    if strongest_area:
        strength_comments = {
            "grammar": "Your essay demonstrates good command of grammar and mechanics.",
            "structure": "The sentence structure in your essay is particularly effective.",
            "coherence": "Your essay is well-organized with good logical flow between ideas.",
            "vocabulary": "You've used a rich and varied vocabulary throughout your essay."
        }
        summary += strength_comments.get(strongest_area, "")
    
    # Add a comment about weakest area
    if weakest_area:
        weakness_comments = {
            "grammar": "However, there are some grammar and spelling issues that need attention.",
            "structure": "However, improving your sentence structure would make your writing more effective.",
            "coherence": "However, the organization and flow of ideas could be improved.",
            "vocabulary": "However, your vocabulary usage could be more varied and precise."
        }
        summary += " " + weakness_comments.get(weakest_area, "")
    
    return summary

def generate_priority_improvements(grammar_results, structure_results, coherence_results, vocabulary_results, score_results):
    """
    Generate a list of priority improvements based on all analyses
    
    Args:
        grammar_results, structure_results, coherence_results, vocabulary_results, score_results: Analysis results
        
    Returns:
        list: Priority improvements
    """
    # Get the weakest area
    weakest_area = score_results.get("weakest_area")
    
    # Collect all suggestions
    all_suggestions = []
    all_suggestions.extend(grammar_results.get("suggestions", []))
    all_suggestions.extend(structure_results.get("suggestions", []))
    all_suggestions.extend(coherence_results.get("suggestions", []))
    all_suggestions.extend(vocabulary_results.get("suggestions", []))
    
    # Prioritize improvements from the weakest area
    priority_improvements = []
    
    # First add suggestions from the weakest area
    if weakest_area == "grammar":
        priority_improvements.extend(grammar_results.get("suggestions", [])[:3])
    elif weakest_area == "structure":
        priority_improvements.extend(structure_results.get("suggestions", [])[:3])
    elif weakest_area == "coherence":
        priority_improvements.extend(coherence_results.get("suggestions", [])[:3])
    elif weakest_area == "vocabulary":
        priority_improvements.extend(vocabulary_results.get("suggestions", [])[:3])
    
    # Then add other high-priority suggestions
    remaining_count = 5 - len(priority_improvements)
    if remaining_count > 0:
        # Get suggestions from other areas, shuffled to avoid always picking the same ones
        other_suggestions = [s for s in all_suggestions if s not in priority_improvements]
        random.shuffle(other_suggestions)
        priority_improvements.extend(other_suggestions[:remaining_count])
    
    return priority_improvements

def generate_highlighted_text(essay_text, issues):
    """
    Generate highlighted text with issues marked
    
    Args:
        essay_text (str): The original essay text
        issues (list): List of grammar issues
        
    Returns:
        list: Segments of text with highlighting information
    """
    # Sort issues by their offset
    sorted_issues = sorted(issues, key=lambda x: x.get("offset", 0))
    
    # Create highlighted segments
    highlighted_segments = []
    last_end = 0
    
    for issue in sorted_issues:
        offset = issue.get("offset", 0)
        length = issue.get("length", 0)
        
        # Add the text before this issue
        if offset > last_end:
            highlighted_segments.append({
                "text": essay_text[last_end:offset],
                "highlighted": False
            })
        
        # Add the highlighted issue
        highlighted_segments.append({
            "text": essay_text[offset:offset+length],
            "highlighted": True,
            "issue": {
                "message": issue.get("message", ""),
                "replacements": issue.get("replacements", [])
            }
        })
        
        last_end = offset + length
    
    # Add the remaining text
    if last_end < len(essay_text):
        highlighted_segments.append({
            "text": essay_text[last_end:],
            "highlighted": False
        })
    
    return highlighted_segments
