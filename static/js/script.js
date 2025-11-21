// Global state
let selectedMode = 'polite';

// DOM Elements
const inputText = document.getElementById('inputText');
const convertBtn = document.getElementById('convertBtn');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const modeButtons = document.querySelectorAll('.mode-btn');
const originalTextDiv = document.getElementById('originalText');
const convertedTextDiv = document.getElementById('convertedText');
const convertedModeTitle = document.getElementById('convertedModeTitle');
const alternativeSuggestions = document.getElementById('alternativeSuggestions');
const datasetInfo = document.getElementById('datasetInfo');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Mode selection
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            modeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedMode = btn.dataset.mode;
        });
    });

    // Convert button
    convertBtn.addEventListener('click', handleConvert);

    // Copy buttons (event delegation)
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('copy-btn') || e.target.closest('.copy-btn')) {
            const btn = e.target.classList.contains('copy-btn') ? e.target : e.target.closest('.copy-btn');
            handleCopy(btn);
        }
    });

    // Enter key to convert
    inputText.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handleConvert();
        }
    });
}

// Handle text conversion
async function handleConvert() {
    const text = inputText.value.trim();
    
    // Validation
    if (!text) {
        showError('Please enter some text to convert!');
        return;
    }

    if (text.length < 3) {
        showError('Please enter at least 3 characters!');
        return;
    }

    // Hide error and results
    hideError();
    resultsSection.style.display = 'none';

    // Show loading state
    setLoadingState(true);

    try {
        // Make API request
        const response = await fetch('/api/convert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                mode: selectedMode
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError('Failed to convert text. Please try again or check if the server is running.');
    } finally {
        setLoadingState(false);
    }
}

// Display conversion results
function displayResults(data) {
    // Original text
    originalTextDiv.textContent = data.original_text;

    // Converted text
    convertedTextDiv.textContent = data.converted_text;
    convertedModeTitle.textContent = `Converted Text (${capitalizeFirst(data.mode)})`;

    // Alternative suggestions
    if (data.alternatives && data.alternatives.length > 0) {
        alternativeSuggestions.innerHTML = data.alternatives.map(alt => `
            <div class="suggestion-item">
                <div class="suggestion-label">${alt.mode}</div>
                <div class="suggestion-text">${escapeHtml(alt.text)}</div>
            </div>
        `).join('');
    } else {
        alternativeSuggestions.innerHTML = '<p style="color: var(--text-secondary);">No alternative suggestions available.</p>';
    }

    // Dataset information
    if (data.dataset_info) {
        const notes = data.dataset_info.notes || [];
        const notesHTML = notes.length > 0 
            ? notes.map(note => `<div class="info-item">${escapeHtml(note)}</div>`).join('')
            : '';
        
        datasetInfo.innerHTML = `
            <div class="info-item"><strong>Model:</strong> ${data.dataset_info.model}</div>
            <div class="info-item"><strong>Datasets Used:</strong> ${data.dataset_info.datasets.join(', ')}</div>
            <div class="info-item"><strong>Processing Time:</strong> ${data.dataset_info.processing_time}</div>
            <div class="info-item"><strong>Confidence:</strong> ${data.dataset_info.confidence}</div>
            <div class="info-item"><strong>Original Sentiment:</strong> ${capitalizeFirst(data.dataset_info.sentiment || 'neutral')}</div>
            <div class="info-item"><strong>Transformation:</strong> ${data.dataset_info.complexity || 'Moderate'}</div>
            ${notesHTML}
        `;
    }

    // Show results with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Handle copy to clipboard
async function handleCopy(button) {
    const target = button.dataset.target;
    const textElement = document.getElementById(target);
    const text = textElement.textContent;

    try {
        await navigator.clipboard.writeText(text);
        
        // Visual feedback
        const originalText = button.textContent;
        button.textContent = '✓ Copied!';
        button.style.background = 'var(--success-color)';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);
    } catch (error) {
        console.error('Failed to copy:', error);
        showError('Failed to copy to clipboard');
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Set loading state
function setLoadingState(isLoading) {
    const btnText = convertBtn.querySelector('.btn-text');
    const loader = convertBtn.querySelector('.loader');
    
    if (isLoading) {
        convertBtn.disabled = true;
        btnText.textContent = 'Converting...';
        loader.style.display = 'inline-block';
    } else {
        convertBtn.disabled = false;
        btnText.textContent = 'Convert Text';
        loader.style.display = 'none';
    }
}

// Utility functions
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Service Worker Registration (optional, for PWA)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment to enable service worker
        // navigator.serviceWorker.register('/service-worker.js');
    });
}
