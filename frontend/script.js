// Global state management
let currentResults = {};
let historyData = [];
let currentTheme = localStorage.getItem('theme') || 'light';
let isAnalyzing = false;
let batchProgress = { current: 0, total: 0 };

// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeEventListeners();
    loadHistory();
    updateStats();
});

// Theme management
function initializeTheme() {
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeToggle();
}

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    updateThemeToggle();
}

function updateThemeToggle() {
    const themeBtn = document.querySelector('.theme-btn i');
    themeBtn.className = currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Event listeners initialization
function initializeEventListeners() {
    // Theme toggle
    document.querySelector('.theme-btn').addEventListener('click', toggleTheme);

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Input method buttons
    document.querySelectorAll('.analyze-btn').forEach(btn => {
        btn.addEventListener('click', handleAnalyzeClick);
    });

    // File upload
    const fileInput = document.getElementById('videoFile');
    const fileUploadArea = document.querySelector('.file-upload-area');

    fileInput.addEventListener('change', handleFileSelect);
    fileUploadArea.addEventListener('click', () => fileInput.click());
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('drop', handleFileDrop);

    // Batch processing
    document.getElementById('batchUrls').addEventListener('input', updateBatchCount);
    document.getElementById('startBatch').addEventListener('click', startBatchAnalysis);

    // Filter controls
    document.querySelector('.filter-toggle').addEventListener('click', toggleFilters);
    document.getElementById('applyFilters').addEventListener('click', applyFilters);
    document.getElementById('clearFilters').addEventListener('click', clearFilters);

    // History actions
    document.getElementById('selectAll').addEventListener('change', toggleSelectAll);
    document.getElementById('exportSelected').addEventListener('click', exportSelected);
    document.getElementById('deleteSelected').addEventListener('click', deleteSelected);

    // Scrape functionality
    document.getElementById('scrapeBtn').addEventListener('click', scrapeVideos);

    // Results actions
    document.getElementById('clearResults').addEventListener('click', clearResults);
    document.getElementById('downloadResults').addEventListener('click', downloadResults);
    document.getElementById('saveResults').addEventListener('click', saveResults);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

// Tab switching
function switchTab(tabId) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');

    // Reset forms when switching tabs
    resetForms();
}

// Handle analyze button clicks
function handleAnalyzeClick(e) {
    e.preventDefault();
    const tabId = e.target.closest('.tab-content').id;

    switch(tabId) {
        case 'youtube':
            analyzeYouTubeVideo();
            break;
        case 'upload':
            analyzeUploadedVideo();
            break;
        case 'batch':
            startBatchAnalysis();
            break;
    }
}

// YouTube video analysis
async function analyzeYouTubeVideo() {
    const url = document.getElementById('youtubeUrl').value.trim();
    if (!url) {
        showMessage('Please enter a YouTube URL', 'error');
        return;
    }

    if (!isValidYouTubeUrl(url)) {
        showMessage('Please enter a valid YouTube URL', 'error');
        return;
    }

    await analyzeVideo('youtube', null, { youtube_url: url });
}

// File upload handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndDisplayFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        document.getElementById('videoFile').files = files;
        validateAndDisplayFile(file);
    }
}

function validateAndDisplayFile(file) {
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/flv', 'video/webm'];
    const maxSize = 500 * 1024 * 1024; // 500MB

    if (!allowedTypes.includes(file.type)) {
        showMessage('Please select a valid video file (MP4, AVI, MOV, WMV, FLV, WebM)', 'error');
        return;
    }

    if (file.size > maxSize) {
        showMessage('File size must be less than 500MB', 'error');
        return;
    }

    displayFileInfo(file);
}

function displayFileInfo(file) {
    const fileInfo = document.querySelector('.file-info');
    const fileDetails = fileInfo.querySelector('.file-details');

    fileDetails.innerHTML = `
        <div class="file-details">
            <i class="fas fa-video"></i>
            <span>${file.name}</span>
            <span>${formatFileSize(file.size)}</span>
        </div>
    `;

    fileInfo.style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function analyzeUploadedVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];

    if (!file) {
        showMessage('Please select a video file', 'error');
        return;
    }

    analyzeVideo('file', file);
}

// Batch analysis
function updateBatchCount() {
    const textarea = document.getElementById('batchUrls');
    const urls = textarea.value.trim().split('\n').filter(url => url.trim());
    const count = urls.length;

    document.getElementById('batchCount').textContent = `${count} video${count !== 1 ? 's' : ''}`;
}

async function startBatchAnalysis() {
    const urls = document.getElementById('batchUrls').value.trim().split('\n').filter(url => url.trim());

    if (urls.length === 0) {
        showMessage('Please enter at least one YouTube URL', 'error');
        return;
    }

    if (urls.length > 10) {
        showMessage('Maximum 10 videos allowed for batch processing', 'error');
        return;
    }

    // Validate URLs
    const invalidUrls = urls.filter(url => !isValidYouTubeUrl(url));
    if (invalidUrls.length > 0) {
        showMessage('Some URLs are invalid. Please check and try again.', 'error');
        return;
    }

    batchProgress = { current: 0, total: urls.length };
    showBatchProgress();

    for (let i = 0; i < urls.length; i++) {
        batchProgress.current = i + 1;
        updateBatchProgress();

        try {
            await analyzeVideo('youtube', null, { youtube_url: urls[i] }, false);
            await new Promise(resolve => setTimeout(resolve, 1000)); // Rate limiting
        } catch (error) {
            console.error(`Failed to analyze video ${i + 1}:`, error);
        }
    }

    hideBatchProgress();
    showMessage(`Batch analysis completed! ${urls.length} videos processed.`, 'success');
    loadHistory();
}

// Main analysis function
async function analyzeVideo(type, file = null, params = {}, showLoading = true) {
    if (isAnalyzing) return;

    isAnalyzing = true;

    try {
        if (showLoading) {
            showLoadingOverlay('Analyzing video... This may take 10-30 seconds');
        }

        let formData = new FormData();
        formData.append('action', 'analyze');

        if (type === 'youtube') {
            formData.append('youtube_url', params.youtube_url);
            formData.append('input_method', 'youtube');
        } else if (type === 'file' && file) {
            formData.append('video_file', file);
            formData.append('input_method', 'manual');
        }

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);
        updateStats();

        if (showLoading) {
            showMessage('Analysis completed successfully!', 'success');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        showMessage(error.message || 'Analysis failed. Please try again.', 'error');
    } finally {
        isAnalyzing = false;
        if (showLoading) {
            hideLoadingOverlay();
        }
    }
}

// Display results
function displayResults(result) {
    currentResults = result;

    // Update metrics grid
    updateEmotionCard(result);
    updateGenderCard(result);
    updateActionsCard(result);
    updateAttributesCard(result);

    // Update description
    updateDescriptionCard(result);

    // Show results section
    document.getElementById('resultsSection').style.display = 'block';

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

    // Add to history
    addToHistory(result);
}

function updateEmotionCard(result) {
    const emotionCard = document.querySelector('.emotion-card');
    const emotionLabel = emotionCard.querySelector('.emotion-label');
    const confidenceBar = emotionCard.querySelector('.confidence-fill');
    const confidenceText = emotionCard.querySelector('.confidence-text');

    if (result.emotion && result.confidence) {
        emotionLabel.textContent = result.emotion;
        confidenceBar.style.width = `${result.confidence * 100}%`;
        confidenceText.textContent = `${Math.round(result.confidence * 100)}%`;
    } else {
        emotionLabel.textContent = 'Not detected';
        confidenceBar.style.width = '0%';
        confidenceText.textContent = 'N/A';
    }
}

function updateGenderCard(result) {
    const genderCard = document.querySelector('.gender-card');
    const demoInfo = genderCard.querySelector('.demo-info');

    demoInfo.innerHTML = `
        <div class="demo-item">
            <span class="demo-label">Gender</span>
            <span class="demo-value">${result.gender || 'Not detected'}</span>
        </div>
        <div class="demo-item">
            <span class="demo-label">Age Range</span>
            <span class="demo-value">${result.age_range || 'Not detected'}</span>
        </div>
    `;
}

function updateActionsCard(result) {
    const actionsCard = document.querySelector('.actions-card');
    const actionsList = actionsCard.querySelector('.actions-list');

    if (result.actions && result.actions.length > 0) {
        actionsList.innerHTML = result.actions.map(action =>
            `<span>${action}</span>`
        ).join('');
    } else {
        actionsList.innerHTML = '<span class="no-data">No actions detected</span>';
    }
}

function updateAttributesCard(result) {
    const attributesCard = document.querySelector('.attributes-card');
    const attributesList = attributesCard.querySelector('.attributes-list');

    if (result.attributes && result.attributes.length > 0) {
        attributesList.innerHTML = result.attributes.map(attr =>
            `<span>${attr}</span>`
        ).join('');
    } else {
        attributesList.innerHTML = '<span class="no-data">No attributes detected</span>';
    }
}

function updateDescriptionCard(result) {
    const descriptionCard = document.querySelector('.description-card');
    const descriptionText = descriptionCard.querySelector('#descriptionText');
    const metaItems = descriptionCard.querySelectorAll('.meta-item');

    // Update description text
    descriptionText.textContent = result.description || 'No description available';

    // Update meta information
    const metaData = [
        { icon: 'fas fa-language', label: 'Language', value: result.language },
        { icon: 'fas fa-globe', label: 'Region', value: result.region },
        { icon: 'fas fa-shield-alt', label: 'Ethical Score', value: result.ethical_score ? `${Math.round(result.ethical_score * 100)}%` : null },
        { icon: 'fas fa-clock', label: 'Duration', value: result.duration ? `${Math.round(result.duration)}s` : null }
    ];

    metaItems.forEach((item, index) => {
        const data = metaData[index];
        if (data && data.value) {
            item.innerHTML = `<i class="${data.icon}"></i> ${data.label}: ${data.value}`;
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
}

// History management
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history`);
        if (!response.ok) throw new Error('Failed to load history');

        historyData = await response.json();
        updateHistoryTable();
        updateHistoryStats();
    } catch (error) {
        console.error('Failed to load history:', error);
        showMessage('Failed to load analysis history', 'error');
    }
}

function addToHistory(result) {
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        video_id: result.video_id || 'Unknown',
        input_method: result.input_method || 'manual',
        emotion: result.emotion || 'Unknown',
        gender: result.gender || 'Unknown',
        age_range: result.age_range || 'Unknown',
        actions: result.actions || [],
        attributes: result.attributes || [],
        language: result.language || 'Unknown',
        region: result.region || 'Unknown',
        ethical_score: result.ethical_score || 0,
        confidence: result.confidence || 0
    };

    historyData.unshift(historyItem);
    updateHistoryTable();
}

function updateHistoryTable() {
    const tbody = document.getElementById('historyTableBody');
    tbody.innerHTML = '';

    if (historyData.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="8">
                    <div class="empty-state">
                        <i class="fas fa-history"></i>
                        <h4>No analysis history</h4>
                        <p>Start by analyzing your first video</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    historyData.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><input type="checkbox" class="row-checkbox" data-id="${item.id}"></td>
            <td>${item.video_id}</td>
            <td><span class="badge ${item.input_method === 'youtube' ? 'info' : 'secondary'}">${item.input_method}</span></td>
            <td><span class="badge success">${item.emotion}</span></td>
            <td>${item.gender}</td>
            <td>${item.age_range}</td>
            <td>${item.language}</td>
            <td>${new Date(item.timestamp).toLocaleString()}</td>
        `;
        tbody.appendChild(row);
    });
}

function updateHistoryStats() {
    const stats = {
        total: historyData.length,
        youtube: historyData.filter(item => item.input_method === 'youtube').length,
        manual: historyData.filter(item => item.input_method === 'manual').length,
        today: historyData.filter(item => {
            const today = new Date().toDateString();
            return new Date(item.timestamp).toDateString() === today;
        }).length
    };

    document.getElementById('totalAnalyses').textContent = stats.total;
    document.getElementById('youtubeAnalyses').textContent = stats.youtube;
    document.getElementById('manualAnalyses').textContent = stats.manual;
    document.getElementById('todayAnalyses').textContent = stats.today;
}

// Filter functionality
function toggleFilters() {
    const filterControls = document.querySelector('.filter-controls');
    const toggleBtn = document.querySelector('.filter-toggle i');

    filterControls.classList.toggle('show');
    toggleBtn.className = filterControls.classList.contains('show') ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
}

async function applyFilters() {
    const filters = {
        emotion: document.getElementById('emotionFilter').value,
        gender: document.getElementById('genderFilter').value,
        age_range: document.getElementById('ageFilter').value,
        region: document.getElementById('regionFilter').value,
        language: document.getElementById('languageFilter').value,
        input_method: document.getElementById('methodFilter').value
    };

    try {
        const queryParams = new URLSearchParams();
        Object.entries(filters).forEach(([key, value]) => {
            if (value) queryParams.append(key, value);
        });

        const response = await fetch(`${API_BASE_URL}/filter?${queryParams}`);
        if (!response.ok) throw new Error('Failed to apply filters');

        const filteredData = await response.json();
        historyData = filteredData;
        updateHistoryTable();
    } catch (error) {
        console.error('Filter error:', error);
        showMessage('Failed to apply filters', 'error');
    }
}

function clearFilters() {
    document.querySelectorAll('.filter-group select').forEach(select => {
        select.value = '';
    });
    loadHistory();
}

// Bulk actions
function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('selectAll');
    const checkboxes = document.querySelectorAll('.row-checkbox');

    checkboxes.forEach(checkbox => {
        checkbox.checked = selectAllCheckbox.checked;
    });
}

async function exportSelected() {
    const selectedIds = Array.from(document.querySelectorAll('.row-checkbox:checked'))
        .map(checkbox => parseInt(checkbox.dataset.id));

    if (selectedIds.length === 0) {
        showMessage('Please select items to export', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/export_selected`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: selectedIds })
        });

        if (!response.ok) throw new Error('Export failed');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `selected_analyses_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);

        showMessage('Export completed successfully!', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showMessage('Failed to export selected items', 'error');
    }
}

async function deleteSelected() {
    const selectedIds = Array.from(document.querySelectorAll('.row-checkbox:checked'))
        .map(checkbox => parseInt(checkbox.dataset.id));

    if (selectedIds.length === 0) {
        showMessage('Please select items to delete', 'error');
        return;
    }

    if (!confirm(`Are you sure you want to delete ${selectedIds.length} item(s)?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/delete_selected`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: selectedIds })
        });

        if (!response.ok) throw new Error('Delete failed');

        historyData = historyData.filter(item => !selectedIds.includes(item.id));
        updateHistoryTable();
        updateHistoryStats();

        showMessage('Selected items deleted successfully!', 'success');
    } catch (error) {
        console.error('Delete error:', error);
        showMessage('Failed to delete selected items', 'error');
    }
}

// Scrape functionality
async function scrapeVideos() {
    const query = document.getElementById('scrapeQuery').value.trim();
    const count = parseInt(document.getElementById('scrapeCount').value);

    if (!query) {
        showMessage('Please enter a search query', 'error');
        return;
    }

    if (count < 1 || count > 20) {
        showMessage('Please enter a count between 1 and 20', 'error');
        return;
    }

    try {
        showLoadingOverlay('Searching and analyzing videos... This may take several minutes');

        const response = await fetch(`${API_BASE_URL}/scrape`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, max_results: count })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const result = await response.json();
        showMessage(result.message || 'Scraping completed successfully!', 'success');
        loadHistory();
        updateStats();

    } catch (error) {
        console.error('Scraping error:', error);
        showMessage(error.message || 'Scraping failed. Please try again.', 'error');
    } finally {
        hideLoadingOverlay();
    }
}

// Results actions
function clearResults() {
    document.getElementById('resultsSection').style.display = 'none';
    currentResults = {};
}

async function downloadResults() {
    if (!currentResults || Object.keys(currentResults).length === 0) {
        showMessage('No results to download', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/download_results`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) throw new Error('Download failed');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis_result_${Date.now()}.json`;
        a.click();
        window.URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Download error:', error);
        showMessage('Failed to download results', 'error');
    }
}

async function saveResults() {
    if (!currentResults || Object.keys(currentResults).length === 0) {
        showMessage('No results to save', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/save_results`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResults)
        });

        if (!response.ok) throw new Error('Save failed');

        showMessage('Results saved successfully!', 'success');
        loadHistory();

    } catch (error) {
        console.error('Save error:', error);
        showMessage('Failed to save results', 'error');
    }
}

// Loading and progress management
function showLoadingOverlay(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    overlay.querySelector('.loading-text h3').textContent = message;
    overlay.style.display = 'flex';
}

function hideLoadingOverlay() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showBatchProgress() {
    const progressContainer = document.querySelector('.batch-progress');
    progressContainer.style.display = 'block';
    updateBatchProgress();
}

function updateBatchProgress() {
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');

    const percentage = (batchProgress.current / batchProgress.total) * 100;
    progressBar.style.width = `${percentage}%`;
    progressText.innerHTML = `
        <span>Processing video ${batchProgress.current} of ${batchProgress.total}</span>
        <span>${Math.round(percentage)}%</span>
    `;
}

function hideBatchProgress() {
    document.querySelector('.batch-progress').style.display = 'none';
}

// Message system
function showMessage(message, type = 'info') {
    const overlay = document.createElement('div');
    overlay.className = `message-overlay ${type}`;
    overlay.innerHTML = `
        <div class="message-container ${type}">
            <div class="message-icon">
                <i class="${getMessageIcon(type)}"></i>
            </div>
            <h3 class="message-title">${getMessageTitle(type)}</h3>
            <p class="message-text">${message}</p>
            <button class="message-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    document.body.appendChild(overlay);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (overlay.parentElement) {
            overlay.remove();
        }
    }, 5000);
}

function getMessageIcon(type) {
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-triangle',
        warning: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle'
    };
    return icons[type] || icons.info;
}

function getMessageTitle(type) {
    const titles = {
        success: 'Success!',
        error: 'Error',
        warning: 'Warning',
        info: 'Information'
    };
    return titles[type] || titles.info;
}

// Utility functions
function resetForms() {
    document.getElementById('youtubeUrl').value = '';
    document.getElementById('videoFile').value = '';
    document.querySelector('.file-info').style.display = 'none';
    document.getElementById('batchUrls').value = '';
    document.getElementById('batchCount').textContent = '0 videos';
    document.getElementById('scrapeQuery').value = '';
}

function isValidYouTubeUrl(url) {
    const patterns = [
        /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/,
        /(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^"&?\/\s]{11})/,
        /(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^"&?\/\s]{11})/
    ];

    return patterns.some(pattern => pattern.test(url));
}

function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('youtubeUrl').focus();
    }

    // Escape to clear results
    if (e.key === 'Escape') {
        clearResults();
    }
}

async function updateStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        if (!response.ok) return;

        const stats = await response.json();

        // Update dashboard stats
        document.getElementById('totalVideos').textContent = stats.total || 0;
        document.getElementById('avgConfidence').textContent = stats.avg_confidence ? `${Math.round(stats.avg_confidence * 100)}%` : '0%';
        document.getElementById('topEmotion').textContent = stats.top_emotion || 'None';
        document.getElementById('analysisRate').textContent = stats.analysis_rate || '0/day';

    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeEventListeners();
    loadHistory();
    updateStats();
});
