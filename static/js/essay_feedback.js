/**
 * Automated Essay Feedback System - Client-side JavaScript
 * Handles UI interactions and feedback display
 */

// Initialize variables
let scoreChart;

// DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // UI Element references
    const essayInput = document.getElementById('essayInput');
    const analyzeButton = document.getElementById('analyzeButton');
    const clearButton = document.getElementById('clearButton');
    const wordCount = document.getElementById('wordCount');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const sampleEssaysDropdown = document.getElementById('sampleEssaysDropdown');

    // Word count update
    essayInput.addEventListener('input', function() {
        updateWordCount(this.value);
    });

    // Analyze button
    analyzeButton.addEventListener('click', function() {
        const essayText = essayInput.value.trim();
        
        if (essayText.length === 0) {
            showAlert('Please enter an essay to analyze.', 'danger');
            return;
        }
        
        analyzeEssay(essayText);
    });

    // Clear button
    clearButton.addEventListener('click', function() {
        essayInput.value = '';
        updateWordCount('');
        resultsSection.classList.add('d-none');
    });

    // Load sample essays
    loadSampleEssays();
    
    // Initialize chart with placeholder data
    initScoreChart();
});

/**
 * Updates the word count display
 * @param {string} text - The essay text
 */
function updateWordCount(text) {
    const words = text.trim().split(/\s+/).filter(word => word.length > 0);
    const count = words.length;
    document.getElementById('wordCount').textContent = `${count} words`;
}

/**
 * Shows an alert message
 * @param {string} message - The message to display
 * @param {string} type - The Bootstrap alert type (success, danger, etc.)
 */
function showAlert(message, type = 'info') {
    // Create alert element
    const alertEl = document.createElement('div');
    alertEl.className = `alert alert-${type} alert-dismissible fade show`;
    alertEl.role = 'alert';
    alertEl.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Find the container to add it to
    const container = document.querySelector('main > .container');
    container.insertBefore(alertEl, container.firstChild);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        const bsAlert = new bootstrap.Alert(alertEl);
        bsAlert.close();
    }, 5000);
}

/**
 * Load sample essays from the server
 */
function loadSampleEssays() {
    fetch('/sample_essays')
        .then(response => response.json())
        .then(samples => {
            // Clear loading placeholder
            document.getElementById('sampleEssaysDropdown').innerHTML = '';
            
            // Add each sample to dropdown
            samples.forEach((sample, index) => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.className = 'dropdown-item sample-essay-item';
                a.href = '#';
                a.textContent = sample.title;
                a.dataset.essayIndex = index;
                a.addEventListener('click', function(e) {
                    e.preventDefault();
                    document.getElementById('essayInput').value = sample.text;
                    updateWordCount(sample.text);
                });
                li.appendChild(a);
                document.getElementById('sampleEssaysDropdown').appendChild(li);
            });
        })
        .catch(error => {
            console.error('Error loading sample essays:', error);
            document.getElementById('sampleEssaysDropdown').innerHTML = 
                '<li><a class="dropdown-item" href="#">Error loading samples</a></li>';
        });
}

/**
 * Send essay to server for analysis
 * @param {string} essayText - The essay text to analyze
 */
function analyzeEssay(essayText) {
    // Show loading spinner, hide results
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    
    loadingSpinner.classList.remove('d-none');
    resultsSection.classList.add('d-none');
    
    // Animate progress bar
    const progressBar = loadingSpinner.querySelector('.progress-bar');
    progressBar.style.width = '0%';
    
    // Send request to server
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ essay: essayText }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(feedback => {
        // Hide loading spinner
        loadingSpinner.classList.add('d-none');
        
        // Display feedback
        displayFeedback(feedback);
        
        // Show results section
        resultsSection.classList.remove('d-none');
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error analyzing essay:', error);
        loadingSpinner.classList.add('d-none');
        showAlert('Error analyzing essay: ' + error.message, 'danger');
    });
}

/**
 * Initialize the score chart with placeholder data
 */
function initScoreChart() {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    
    scoreChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Grammar', 'Structure', 'Coherence', 'Vocabulary'],
            datasets: [{
                label: 'Score',
                data: [0, 0, 0, 0],
                fill: true,
                backgroundColor: 'rgba(13, 110, 253, 0.2)',
                borderColor: 'rgba(13, 110, 253, 1)',
                pointBackgroundColor: 'rgba(13, 110, 253, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(13, 110, 253, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 10
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Display feedback results in the UI
 * @param {Object} feedback - The feedback data from the server
 */
function displayFeedback(feedback) {
    // Check for errors
    if (feedback.error) {
        showAlert('Error: ' + feedback.error, 'danger');
        return;
    }
    
    // Overall assessment
    const overallAssessment = feedback.overall_assessment;
    document.getElementById('overallScore').textContent = overallAssessment.score.toFixed(1);
    document.getElementById('overallGrade').textContent = overallAssessment.grade;
    document.getElementById('feedbackSummary').textContent = overallAssessment.summary;
    
    // Update chart data
    updateScoreChart(overallAssessment.score_breakdown);
    
    // Priority improvements
    displayPriorityImprovements(feedback.priority_improvements);
    
    // Grammar & spelling
    const grammarFeedback = feedback.grammar_feedback;
    document.getElementById('grammarScore').textContent = grammarFeedback.score.toFixed(1);
    document.getElementById('grammarBadge').textContent = grammarFeedback.total_issues || 0;
    
    // Grammar suggestions
    displaySuggestions('grammarSuggestions', grammarFeedback.suggestions);
    
    // Grammar issues table
    const grammarIssuesTable = document.getElementById('grammarIssuesTable');
    if (grammarFeedback.issues && grammarFeedback.issues.length > 0) {
        let tableHTML = '';
        grammarFeedback.issues.forEach(issue => {
            tableHTML += `
                <tr>
                    <td>${issue.message}</td>
                    <td class="text-muted"><small>"...${issue.sentence}..."</small></td>
                    <td>${issue.replacements.length > 0 ? issue.replacements.join(', ') : 'N/A'}</td>
                </tr>
            `;
        });
        grammarIssuesTable.innerHTML = tableHTML;
    } else {
        grammarIssuesTable.innerHTML = '<tr><td colspan="3" class="text-center">No grammar issues detected</td></tr>';
    }
    
    // Sentence structure
    const structureFeedback = feedback.structure_feedback;
    document.getElementById('structureScore').textContent = structureFeedback.score.toFixed(1);
    document.getElementById('avgSentenceLength').textContent = structureFeedback.metrics.avg_sentence_length.toFixed(1);
    document.getElementById('readabilityScore').textContent = structureFeedback.metrics.flesch_reading_ease.toFixed(1);
    
    // Structure suggestions
    displaySuggestions('structureSuggestions', structureFeedback.suggestions);
    
    // Coherence
    const coherenceFeedback = feedback.coherence_feedback;
    document.getElementById('coherenceScore').textContent = coherenceFeedback.score.toFixed(1);
    
    // Introduction and conclusion indicators
    const hasIntroElement = document.getElementById('hasIntroduction');
    hasIntroElement.innerHTML = coherenceFeedback.has_introduction ? 
        '<i class="fas fa-check-circle text-success"></i>' : 
        '<i class="fas fa-times-circle text-danger"></i>';
    
    const hasConclElement = document.getElementById('hasConclusion');
    hasConclElement.innerHTML = coherenceFeedback.has_conclusion ? 
        '<i class="fas fa-check-circle text-success"></i>' : 
        '<i class="fas fa-times-circle text-danger"></i>';
    
    // Coherence suggestions
    displaySuggestions('coherenceSuggestions', coherenceFeedback.suggestions);
    
    // Vocabulary
    const vocabFeedback = feedback.vocabulary_feedback;
    document.getElementById('vocabularyScore').textContent = vocabFeedback.score.toFixed(1);
    document.getElementById('uniqueWords').textContent = vocabFeedback.metrics.unique_words;
    document.getElementById('totalWords').textContent = vocabFeedback.metrics.total_words;
    document.getElementById('typeTokenRatio').textContent = vocabFeedback.metrics.type_token_ratio.toFixed(2);
    
    // Most common words
    const mostCommonWordsEl = document.getElementById('mostCommonWords');
    if (vocabFeedback.most_common_words && vocabFeedback.most_common_words.length > 0) {
        let wordsHTML = '';
        vocabFeedback.most_common_words.forEach(wordInfo => {
            wordsHTML += `
                <span class="word-frequency">
                    ${wordInfo[0]} <span class="count">${wordInfo[1]}</span>
                </span>
            `;
        });
        mostCommonWordsEl.innerHTML = wordsHTML;
    } else {
        mostCommonWordsEl.innerHTML = 'No word frequency data available';
    }
    
    // Vocabulary suggestions
    displaySuggestions('vocabularySuggestions', vocabFeedback.suggestions);
    
    // Marked text with issues
    if (feedback.highlighted_text) {
        displayHighlightedText(feedback.highlighted_text);
    } else {
        document.getElementById('markedText').innerHTML = '<p>No text highlighting available</p>';
    }
}

/**
 * Update the radar chart with score breakdown
 * @param {Object} scoreBreakdown - The score breakdown object
 */
function updateScoreChart(scoreBreakdown) {
    if (!scoreChart) return;
    
    scoreChart.data.datasets[0].data = [
        scoreBreakdown.grammar,
        scoreBreakdown.structure,
        scoreBreakdown.coherence,
        scoreBreakdown.vocabulary
    ];
    
    scoreChart.update();
}

/**
 * Display priority improvements list
 * @param {Array} improvements - List of improvement suggestions
 */
function displayPriorityImprovements(improvements) {
    const priorityList = document.getElementById('priorityList');
    
    if (!improvements || improvements.length === 0) {
        priorityList.innerHTML = '<li><span class="fa-li"><i class="fas fa-arrow-right"></i></span>No specific improvements needed</li>';
        return;
    }
    
    let listHTML = '';
    improvements.forEach(suggestion => {
        listHTML += `
            <li>
                <span class="fa-li"><i class="fas fa-arrow-right"></i></span>
                ${suggestion}
            </li>
        `;
    });
    
    priorityList.innerHTML = listHTML;
}

/**
 * Display suggestions in a list
 * @param {string} elementId - ID of the list element
 * @param {Array} suggestions - List of suggestions
 */
function displaySuggestions(elementId, suggestions) {
    const listElement = document.getElementById(elementId);
    
    if (!suggestions || suggestions.length === 0) {
        listElement.innerHTML = '<li class="list-group-item">No specific suggestions</li>';
        return;
    }
    
    let listHTML = '';
    suggestions.forEach(suggestion => {
        listHTML += `<li class="list-group-item">${suggestion}</li>`;
    });
    
    listElement.innerHTML = listHTML;
}

/**
 * Display highlighted text with issues
 * @param {Array} segments - Text segments with highlighting information
 */
function displayHighlightedText(segments) {
    const markedTextEl = document.getElementById('markedText');
    let html = '<p>';
    
    segments.forEach(segment => {
        if (segment.highlighted) {
            const message = segment.issue.message;
            const replacements = segment.issue.replacements.length > 0 
                ? `Suggestions: ${segment.issue.replacements.join(', ')}` 
                : 'No specific suggestions';
            
            html += `<span class="highlight" data-message="${message}. ${replacements}">${segment.text}</span>`;
        } else {
            html += segment.text;
        }
    });
    
    html += '</p>';
    markedTextEl.innerHTML = html;
}
