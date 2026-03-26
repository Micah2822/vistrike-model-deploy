// VISTRIKE Model Testing UI - JavaScript

const API_BASE = '';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const videoPreview = document.getElementById('videoPreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const annotatedVideo = document.getElementById('annotatedVideo');
const attributesDisplay = document.getElementById('attributesDisplay');
const statsContent = document.getElementById('statsContent');
const downloadVideoBtn = document.getElementById('downloadVideoBtn');
const downloadAllBtn = document.getElementById('downloadAllBtn');

// Configuration sliders
const confidenceSlider = document.getElementById('confidence');
const confidenceValue = document.getElementById('confidenceValue');
const attrConfidenceSlider = document.getElementById('attrConfidence');
const attrConfidenceValue = document.getElementById('attrConfidenceValue');

let currentVideoFile = null;
let charts = [];
let progressInterval = null;
let currentOutputDir = null;

// Real-time dashboard state
let analysisData = null;  // Loaded from analysis.json
let computedEvents = [];  // Pre-computed action events with frame numbers
let videoFps = 30;        // Video FPS for frame calculation
let totalFrames = 0;      // Total frames in video
let lastDisplayedFrame = -1;  // Track last frame to avoid redundant updates

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateSliderValues();
    checkServerStatus();
});

async function checkServerStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        if (!data.inference_script) {
            showError(
                `Inference script not found at: ${data.inference_script_path}\n\nPlease ensure scripts/10_inference.py exists.`,
                'Configuration Error'
            );
        }
        
        if (!data.has_unified_model) {
            console.warn('No unified model found. Inference may fail.');
            // Don't show error, just warn - user might be testing
        }
    } catch (error) {
        console.error('Could not check server status:', error);
    }
}

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => videoInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-secondary)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-secondary)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Sliders
    confidenceSlider.addEventListener('input', () => {
        confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
    });
    
    attrConfidenceSlider.addEventListener('input', () => {
        attrConfidenceValue.textContent = parseFloat(attrConfidenceSlider.value).toFixed(2);
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeVideo);
}

function updateSliderValues() {
    confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
    attrConfidenceValue.textContent = parseFloat(attrConfidenceSlider.value).toFixed(2);
}

function handleFileSelect(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a video file');
        return;
    }
    
    if (file.size > 500 * 1024 * 1024) {
        alert('File size must be less than 500MB');
        return;
    }
    
    currentVideoFile = file;
    
    // Show preview
    const video = document.createElement('video');
    video.src = URL.createObjectURL(file);
    video.controls = true;
    video.className = 'result-video';
    
    videoPreview.innerHTML = '';
    videoPreview.appendChild(video);
    videoPreview.classList.remove('hidden');
    
    // Enable analyze button
    analyzeBtn.disabled = false;
}

async function analyzeVideo() {
    if (!currentVideoFile) {
        alert('Please select a video file');
        return;
    }
    
    // Show progress
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    // Reset progress
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressText').textContent = 'Uploading video...';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('video', currentVideoFile);
    formData.append('confidence', confidenceSlider.value);
    formData.append('attr_confidence', attrConfidenceSlider.value);
    formData.append('device', document.getElementById('device').value);
    formData.append('save_video', document.getElementById('saveVideo').checked ? 'true' : 'false');
    
    try {
        // Update message to show upload starting
        document.getElementById('progressText').textContent = 'Uploading video...';
        console.log('Starting upload...');
        
        // Make API call with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.error('Upload timeout after 10 minutes');
            controller.abort();
        }, 600000); // 10 minute timeout
        
        let response;
        try {
            console.log('Sending fetch request...');
            response = await fetch(`${API_BASE}/api/analyze`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            console.log('Fetch response received:', response.status);
        } catch (error) {
            clearTimeout(timeoutId);
            console.error('Fetch error:', error);
            if (error.name === 'AbortError') {
                throw new Error('Upload timeout - file may be too large or connection is slow');
            }
            throw error;
        }
        
        // Upload complete, now processing
        document.getElementById('progressText').textContent = 'Video uploaded. Starting analysis...';
        console.log('Parsing response JSON...');
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (!response.ok) {
            stopProgressPolling();
            const errorMsg = data.details || data.error || 'Analysis failed';
            console.error('API error:', errorMsg);
            showError(errorMsg, data.error || 'Error');
            progressSection.classList.add('hidden');
            analyzeBtn.disabled = false;
            return;
        }
        
        // Start polling for progress immediately
        console.log('Starting progress polling for:', data.output_dir);
        currentOutputDir = data.output_dir;
        
        // Show model status section immediately
        document.getElementById('modelStatus').classList.remove('hidden');
        document.getElementById('deviceStatus').textContent = document.getElementById('device').value.toUpperCase();
        
        startProgressPolling(data.output_dir);
        
        // Don't show results yet - wait for completion via progress polling
        // displayResults will be called when progress.status === 'completed'
        
    } catch (error) {
        console.error('Error in analyzeVideo:', error);
        stopProgressPolling();
        showError(error.message, 'Upload Error');
        progressSection.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

function startProgressPolling(outputDir) {
    // Clear any existing interval
    stopProgressPolling();
    
    // Poll immediately, then every 500ms
    const pollProgress = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/progress/${encodeURIComponent(outputDir)}`);
            
            if (!response.ok) {
                // Progress file might not exist yet, that's okay - keep polling
                if (response.status === 404 || response.status === 200) {
                    // Show initializing state
                    const initData = await response.json().catch(() => ({
                        status: 'initializing',
                        message: 'Initializing analysis...',
                        device: 'Unknown'
                    }));
                    updateProgressDisplay(initData);
                    return; // Keep polling
                }
                console.error(`Progress fetch failed: ${response.status}`);
                throw new Error(`Progress fetch failed: ${response.status}`);
            }
            
            const progress = await response.json();
            
            if (progress.status === 'completed' || progress.status === 'failed') {
                stopProgressPolling();
                if (progress.status === 'failed') {
                    document.getElementById('progressText').textContent = 'Analysis failed';
                    progressSection.classList.remove('hidden');
                } else {
                    document.getElementById('progressText').textContent = 'Analysis completed!';
                    // Fetch final results and display
                    fetchFinalResults(outputDir);
                }
                return;
            }
            
            // Update progress display
            updateProgressDisplay(progress);
            
        } catch (error) {
            // Don't log 404s as errors (file doesn't exist yet)
            if (!error.message.includes('404')) {
                console.error('Error fetching progress:', error);
            }
        }
    };
    
    // Poll immediately
    pollProgress();
    
    // Then poll every 500ms
    progressInterval = setInterval(pollProgress, 500);
}

function stopProgressPolling() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

function updateProgressDisplay(progress) {
    console.log('Updating progress display:', progress);
    
    // Always show model status section
    document.getElementById('modelStatus').classList.remove('hidden');
    
    // Update model status
    if (progress.device) {
        document.getElementById('deviceStatus').textContent = progress.device.toUpperCase();
    } else {
        document.getElementById('deviceStatus').textContent = 'Initializing...';
    }
    
    if (progress.unified_model) {
        document.getElementById('unifiedModelStatus').classList.remove('hidden');
        document.getElementById('unifiedModelStatus').querySelector('.status-value').textContent = progress.unified_model;
    }
    
    if (progress.action_models) {
        document.getElementById('actionModelsStatus').classList.remove('hidden');
        document.getElementById('actionModelsStatus').querySelector('.status-value').textContent = progress.action_models;
    }
    
    // Update video info
    if (progress.video_resolution || progress.video_fps || progress.total_frames) {
        document.getElementById('videoInfo').classList.remove('hidden');
        if (progress.video_resolution) {
            document.getElementById('videoResolution').textContent = progress.video_resolution;
        }
        if (progress.video_fps) {
            document.getElementById('videoFps').textContent = progress.video_fps.toFixed(2);
        }
        if (progress.total_frames) {
            document.getElementById('videoTotalFrames').textContent = progress.total_frames;
        }
    }
    
    // Update processing progress
    if (progress.current_frame > 0 || progress.total_frames > 0) {
        document.getElementById('processingProgress').classList.remove('hidden');
        
        // Update progress bar
        if (progress.total_frames > 0) {
            const percent = (progress.current_frame / progress.total_frames) * 100;
            document.getElementById('progressFill').style.width = `${percent}%`;
            document.getElementById('progressPercent').textContent = `${percent.toFixed(1)}%`;
        }
        
        // Update time estimates
        if (progress.fps > 0 && progress.total_frames > 0) {
            const elapsed = progress.current_frame / progress.fps;
            const remaining = (progress.total_frames - progress.current_frame) / progress.fps;
            const elapsedStr = formatTime(elapsed);
            const remainingStr = formatTime(remaining);
            document.getElementById('progressTime').textContent = `${elapsedStr} / ${remainingStr}`;
        }
        
        // Update frame counter
        document.getElementById('currentFrame').textContent = progress.current_frame || 0;
        document.getElementById('totalFrames').textContent = `/ ${progress.total_frames || 0}`;
        
        // Update FPS
        const fps = progress.fps || 0;
        document.getElementById('processingFps').textContent = fps.toFixed(2);
        
        // Update boxes detected
        document.getElementById('boxesDetected').textContent = progress.boxes_detected || 0;
        
        // Update confidence
        const conf = progress.avg_confidence || 0;
        document.getElementById('avgConfidence').textContent = conf.toFixed(2);
    }
    
    // Update detection details
    console.log('Detection details in progress:', {
        detection_boxes: progress.detection_boxes,
        score_range: progress.score_range,
        detection_threshold: progress.detection_threshold,
        boxes_above_threshold: progress.boxes_above_threshold
    });
    
    if (progress.detection_boxes !== undefined || progress.score_range || progress.detection_threshold !== undefined || progress.boxes_above_threshold !== undefined) {
        document.getElementById('detectionDetails').classList.remove('hidden');
        
        const boxesEl = document.getElementById('detectionBoxes');
        const rangeEl = document.getElementById('scoreRange');
        const thresholdEl = document.getElementById('detectionThreshold');
        const aboveThresholdEl = document.getElementById('boxesAboveThreshold');
        
        if (progress.detection_boxes !== undefined && progress.detection_boxes !== null) {
            boxesEl.textContent = progress.detection_boxes;
            console.log('Set detectionBoxes to:', progress.detection_boxes);
        } else {
            boxesEl.textContent = '-';
        }
        
        if (progress.score_range) {
            rangeEl.textContent = progress.score_range;
            console.log('Set scoreRange to:', progress.score_range);
        } else {
            rangeEl.textContent = '-';
        }
        
        if (progress.detection_threshold !== undefined && progress.detection_threshold !== null) {
            thresholdEl.textContent = parseFloat(progress.detection_threshold).toFixed(2);
            console.log('Set detectionThreshold to:', progress.detection_threshold);
        } else {
            thresholdEl.textContent = '-';
        }
        
        if (progress.boxes_above_threshold !== undefined && progress.boxes_above_threshold !== null) {
            aboveThresholdEl.textContent = progress.boxes_above_threshold;
            console.log('Set boxesAboveThreshold to:', progress.boxes_above_threshold);
        } else {
            aboveThresholdEl.textContent = '-';
        }
        
        // Show formatted confidence interval info if available
        if (progress.score_range && progress.detection_threshold !== undefined && progress.boxes_above_threshold !== undefined) {
            const intervalItem = document.getElementById('confidenceIntervalItem');
            const intervalValue = document.getElementById('confidenceInterval');
            if (intervalItem && intervalValue) {
                intervalItem.style.display = 'flex';
                intervalValue.textContent = `${progress.score_range} × ${progress.boxes_above_threshold} boxes`;
            }
        }
    }
    
    // Update message
    if (progress.message) {
        document.getElementById('progressText').textContent = progress.message;
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

async function fetchFinalResults(outputDir) {
    try {
        // Extract video stem from outputDir for later use
        // outputDir format: "video_name_timestamp" or just "video_name"
        const outputDirParts = outputDir.split('_');
        // Remove timestamp suffix (last part if it's a number)
        let videoStem;
        if (outputDirParts.length > 1 && /^\d+$/.test(outputDirParts[outputDirParts.length - 1])) {
            videoStem = outputDirParts.slice(0, -1).join('_');
        } else {
            videoStem = outputDir;
        }
        
        // First check progress file for annotated video info
        let annotatedVideo = null;
        try {
            const progressResponse = await fetch(`${API_BASE}/api/progress/${encodeURIComponent(outputDir)}`);
            if (progressResponse.ok) {
                const progressData = await progressResponse.json();
                if (progressData.annotated_video) {
                    annotatedVideo = progressData.annotated_video;
                    console.log('Found annotated video from progress:', annotatedVideo);
                }
            }
        } catch (e) {
            console.log('Could not get progress data:', e);
        }
        
        // If not in progress, try to find it
        if (!annotatedVideo) {
            const possibleVideos = [
                `${outputDir}/${videoStem}_annotated.mp4`,
                `${outputDir}/annotated.mp4`,
                `${outputDir}/${outputDir}_annotated.mp4`,
            ];
            
            console.log('Searching for annotated video:', possibleVideos);
            
            for (const videoPath of possibleVideos) {
                try {
                    const testResponse = await fetch(`${API_BASE}/api/results/${videoPath}`, { method: 'HEAD' });
                    if (testResponse.ok) {
                        annotatedVideo = videoPath.split('/').pop();
                        console.log('Found annotated video by searching:', annotatedVideo);
                        break;
                    }
                } catch {
                    continue;
                }
            }
        }
        
        // Load summary.json to get final results
        console.log('Fetching summary from:', `${API_BASE}/api/results/${outputDir}/summary.json`);
        const response = await fetch(`${API_BASE}/api/results/${outputDir}/summary.json`);
        if (!response.ok) {
            console.error('Summary fetch failed:', response.status, response.statusText);
            throw new Error(`Could not load summary: ${response.status}`);
        }
        const summary = await response.json();
        console.log('Summary loaded:', summary);
        
        // Display results
        displayResults({
            summary: summary,
            output_dir: outputDir,
            annotated_video: annotatedVideo,
            video_name: videoStem
        });
        
        // Hide progress after showing results
        setTimeout(() => {
            progressSection.classList.add('hidden');
        }, 1000);
        
    } catch (error) {
        console.error('Error fetching final results:', error);
        showError(`Analysis completed but could not load results: ${error.message}`);
    }
}

// Removed simulateProgress - now using real progress updates

function displayResults(data) {
    // Don't hide progress yet - wait for completion
    // progressSection.classList.add('hidden');
    
    // Clear previous charts
    charts.forEach(chart => chart.destroy());
    charts = [];
    
    // Show results (but keep progress visible until video loads)
    resultsSection.classList.remove('hidden');
    
    // Reset video element and remove any previous error messages
    const videoCard = document.querySelector('.video-output-card');
    const existingNoVideoMsg = videoCard.querySelector('.no-video-message');
    if (existingNoVideoMsg) {
        existingNoVideoMsg.remove();
    }
    annotatedVideo.style.display = 'block';
    annotatedVideo.src = '';
    
    // Display annotated video
    if (data.annotated_video) {
        const videoUrl = `${API_BASE}/api/results/${data.output_dir}/${data.annotated_video}`;
        console.log('Setting video URL:', videoUrl);
        
        // Add error handling for video load
        annotatedVideo.onerror = (e) => {
            console.error('Video load failed:', videoUrl, e);
            annotatedVideo.style.display = 'none';
            const errorMsg = document.createElement('div');
            errorMsg.className = 'no-video-message';
            errorMsg.innerHTML = `
                <p>Video could not be loaded.</p>
                <a href="${videoUrl}" class="btn-primary" style="display: inline-block; text-decoration: none; margin-top: 16px;" download>Download Video Instead</a>
            `;
            videoCard.appendChild(errorMsg);
        };
        
        annotatedVideo.onloadstart = () => {
            console.log('Video loading:', videoUrl);
        };
        
        annotatedVideo.onloadeddata = () => {
            console.log('Video loaded successfully');
        };
        
        annotatedVideo.src = videoUrl;
        
        // Set up download button for video
        downloadVideoBtn.href = videoUrl;
        downloadVideoBtn.download = data.annotated_video;
        downloadVideoBtn.style.display = 'inline-flex';
    } else {
        // No annotated video - show message
        annotatedVideo.style.display = 'none';
        const noVideoMsg = document.createElement('div');
        noVideoMsg.className = 'no-video-message';
        noVideoMsg.innerHTML = `
            <p>Annotated video was not generated.</p>
            <p style="font-size: 14px; color: var(--text-muted); margin-top: 8px;">
                This may be because "Save Annotated Video" was disabled, or video generation failed.
            </p>
        `;
        videoCard.appendChild(noVideoMsg);
        downloadVideoBtn.style.display = 'none';
    }
    
    // Set up download all button
    downloadAllBtn.onclick = () => downloadAllResults(data);
    downloadAllBtn.style.display = 'inline-flex';
    
    // Store data for downloads
    window.currentResults = data;
    
    // Display attributes and actions
    displayAttributes(data.summary);
    
    // Display statistics
    displayStatistics(data.summary);
    
    // Re-enable button
    analyzeBtn.disabled = false;
    
    // Stop progress polling
    stopProgressPolling();
    
    // Initialize real-time dashboard
    initRealtimeDashboard(data);
}

function showError(message, title = 'Error') {
    // Create error modal
    const errorModal = document.createElement('div');
    errorModal.className = 'error-modal';
    errorModal.innerHTML = `
        <div class="error-modal-content">
            <div class="error-modal-header">
                <h3>${title}</h3>
                <button class="error-modal-close">&times;</button>
            </div>
            <div class="error-modal-body">
                <pre class="error-message">${message}</pre>
            </div>
            <div class="error-modal-footer">
                <button class="btn-primary" onclick="this.closest('.error-modal').remove()">Close</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(errorModal);
    
    // Close on X click
    errorModal.querySelector('.error-modal-close').onclick = () => errorModal.remove();
    
    // Close on background click
    errorModal.onclick = (e) => {
        if (e.target === errorModal) errorModal.remove();
    };
}

async function downloadAllResults(data) {
    try {
        // Create a zip-like download experience
        // Since we can't create zip in browser easily, we'll download files sequentially
        // or provide links to all files
        
        const files = [
            { name: data.annotated_video, path: `${data.output_dir}/${data.annotated_video}`, type: 'video' },
            { name: 'analysis.json', path: `${data.output_dir}/analysis.json`, type: 'json' },
            { name: 'summary.json', path: `${data.output_dir}/summary.json`, type: 'json' },
        ];
        
        // Download each file
        for (const file of files) {
            if (file.name && file.type !== 'video') {
                const url = `${API_BASE}/api/results/${file.path}`;
                const link = document.createElement('a');
                link.href = url;
                link.download = file.name;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                // Small delay between downloads
                await new Promise(resolve => setTimeout(resolve, 300));
            }
        }
        
        // For video, use the existing download button
        if (data.annotated_video) {
            downloadVideoBtn.click();
        }
        
    } catch (error) {
        alert(`Error downloading files: ${error.message}`);
    }
}

function displayAttributes(summary) {
    attributesDisplay.innerHTML = '';
    
    // Check if summary has the expected structure
    if (!summary || !summary.fighters || !summary.actions) {
        attributesDisplay.innerHTML = '<p style="color: var(--text-muted);">No attribute data available.</p>';
        return;
    }
    
    // Fighter visibility
    const redFighter = summary.fighters.red || { frames_visible: 0 };
    const blueFighter = summary.fighters.blue || { frames_visible: 0 };
    const actions = summary.actions || {};
    
    const attributes = [
        { label: 'Red Corner Visible', value: `${redFighter.frames_visible || 0} frames`, class: 'red' },
        { label: 'Blue Corner Visible', value: `${blueFighter.frames_visible || 0} frames`, class: 'blue' },
    ];
    // Dynamically add action type totals from summary (config-driven)
    for (const [type, data] of Object.entries(actions)) {
        const label = `Total ${type.charAt(0).toUpperCase() + type.slice(1)}`;
        attributes.push({ label, value: data?.count || 0 });
    }
    
    attributes.forEach(attr => {
        const item = document.createElement('div');
        item.className = 'attribute-item';
        item.innerHTML = `
            <div class="attribute-label">${attr.label}</div>
            <div class="attribute-value ${attr.class || ''}">${attr.value}</div>
        `;
        attributesDisplay.appendChild(item);
    });
}

function displayStatistics(summary) {
    statsContent.innerHTML = '';
    
    // Check if summary exists
    if (!summary) {
        statsContent.innerHTML = '<p style="color: var(--text-muted);">No statistics available.</p>';
        return;
    }
    
    // Video Info with defensive checks
    const videoInfo = createStatsSection('Video Information', [
        { label: 'Duration', value: `${(summary.duration_seconds || 0).toFixed(1)}s` },
        { label: 'Total Frames', value: summary.total_frames || 0 },
        { label: 'FPS', value: (summary.fps || 0).toFixed(1) },
        { label: 'Frames with Detections', value: summary.detection?.frames_with_detections || 0 },
    ]);
    statsContent.appendChild(videoInfo);
    
    // Fighter Stats
    if (summary.fighters) {
        const fighterStats = createFighterStatsSection(summary);
        statsContent.appendChild(fighterStats);
    }
    
    // Action Statistics - dynamically generated from summary.actions (config-driven)
    if (summary.actions) {
        for (const [actionType, actionData] of Object.entries(summary.actions)) {
            if (!actionData || actionData.count <= 0) continue;
            
            // Build chart configs from by_* keys in the action data
            const chartConfigs = [];
            for (const [key, value] of Object.entries(actionData)) {
                if (key === 'count') continue;
                if (typeof value === 'object' && value !== null && Object.keys(value).length > 0) {
                    // Convert by_attr_name to a readable label: "by_type" -> "Type", "by_attacker" -> "Attacker"
                    const rawLabel = key.replace(/^by_/, '').replace(/_/g, ' ');
                    const label = rawLabel.charAt(0).toUpperCase() + rawLabel.slice(1);
                    chartConfigs.push({ key, label: `${actionType.charAt(0).toUpperCase() + actionType.slice(1)} ${label}` });
                }
            }
            
            const title = `${actionType.charAt(0).toUpperCase() + actionType.slice(1)} Statistics`;
            const section = createActionStatsSection(title, actionData, chartConfigs);
            statsContent.appendChild(section);
        }
    }
}

function createStatsSection(title, stats) {
    const section = document.createElement('div');
    section.className = 'stats-section';
    
    const h3 = document.createElement('h3');
    h3.textContent = title;
    section.appendChild(h3);
    
    const grid = document.createElement('div');
    grid.className = 'stats-grid';
    
    stats.forEach(stat => {
        const item = document.createElement('div');
        item.className = 'stat-item';
        item.innerHTML = `
            <div class="stat-value">${stat.value}</div>
            <div class="stat-label">${stat.label}</div>
        `;
        grid.appendChild(item);
    });
    
    section.appendChild(grid);
    return section;
}

function createFighterStatsSection(summary) {
    const section = document.createElement('div');
    section.className = 'stats-section';
    
    const h3 = document.createElement('h3');
    h3.textContent = 'Fighter Statistics';
    section.appendChild(h3);
    
    const fighterStats = document.createElement('div');
    fighterStats.className = 'fighter-stats';
    
    // Red Fighter
    const redFighter = summary.fighters?.red || { frames_visible: 0, guards: {}, stances: {} };
    const redCard = createFighterCard('Red Corner', redFighter, 'red');
    fighterStats.appendChild(redCard);
    
    // Blue Fighter
    const blueFighter = summary.fighters?.blue || { frames_visible: 0, guards: {}, stances: {} };
    const blueCard = createFighterCard('Blue Corner', blueFighter, 'blue');
    fighterStats.appendChild(blueCard);
    
    section.appendChild(fighterStats);
    return section;
}

function createFighterCard(title, fighter, corner) {
    const card = document.createElement('div');
    card.className = `fighter-card ${corner}`;
    
    const h4 = document.createElement('h4');
    h4.textContent = title;
    card.appendChild(h4);
    
    const grid = document.createElement('div');
    grid.className = 'fighter-stats-grid';
    
    // Ensure fighter has required properties
    const fighterData = fighter || { frames_visible: 0, guards: {}, stances: {} };
    
    // Frames visible
    const framesItem = document.createElement('div');
    framesItem.className = 'fighter-stat';
    framesItem.innerHTML = `
        <div class="fighter-stat-value">${fighterData.frames_visible || 0}</div>
        <div class="fighter-stat-label">Frames Visible</div>
    `;
    grid.appendChild(framesItem);
    
    // Guards
    const guards = fighterData.guards || {};
    if (Object.keys(guards).length > 0) {
        const guardsItem = document.createElement('div');
        guardsItem.className = 'fighter-stat';
        const topGuard = Object.entries(guards)
            .sort((a, b) => b[1] - a[1])[0];
        guardsItem.innerHTML = `
            <div class="fighter-stat-value">${topGuard[0]}</div>
            <div class="fighter-stat-label">Primary Guard</div>
        `;
        grid.appendChild(guardsItem);
    }
    
    // Stances
    const stances = fighterData.stances || {};
    if (Object.keys(stances).length > 0) {
        const stancesItem = document.createElement('div');
        stancesItem.className = 'fighter-stat';
        const topStance = Object.entries(stances)
            .sort((a, b) => b[1] - a[1])[0];
        stancesItem.innerHTML = `
            <div class="fighter-stat-value">${topStance[0]}</div>
            <div class="fighter-stat-label">Primary Stance</div>
        `;
        grid.appendChild(stancesItem);
    }
    
    card.appendChild(grid);
    return card;
}

function createActionStatsSection(title, actionData, chartConfigs) {
    const section = document.createElement('div');
    section.className = 'stats-section';
    
    const h3 = document.createElement('h3');
    h3.textContent = `${title} - Total: ${actionData.count}`;
    section.appendChild(h3);
    
    // Total count
    const totalItem = document.createElement('div');
    totalItem.className = 'stat-item';
    totalItem.style.marginBottom = '24px';
    totalItem.innerHTML = `
        <div class="stat-value">${actionData.count}</div>
        <div class="stat-label">Total ${title.split(' ')[0]}</div>
    `;
    section.appendChild(totalItem);
    
    // Charts
    chartConfigs.forEach(config => {
        const data = actionData[config.key];
        if (data && Object.keys(data).length > 0) {
            const chartContainer = document.createElement('div');
            chartContainer.className = 'chart-container';
            
            const canvas = document.createElement('canvas');
            chartContainer.appendChild(canvas);
            section.appendChild(chartContainer);
            
            // Create pie chart
            const ctx = canvas.getContext('2d');
            const chart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: Object.keys(data),
                    datasets: [{
                        data: Object.values(data),
                        backgroundColor: [
                            'rgba(37, 99, 235, 0.8)',
                            'rgba(239, 68, 68, 0.8)',
                            'rgba(34, 197, 94, 0.8)',
                            'rgba(251, 191, 36, 0.8)',
                            'rgba(168, 85, 247, 0.8)',
                            'rgba(236, 72, 153, 0.8)',
                        ],
                        borderColor: 'var(--bg-primary)',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: config.label,
                            color: '#FFFFFF',
                            font: {
                                size: 16,
                                weight: '600'
                            }
                        },
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#FFFFFF',
                                padding: 12,
                                font: {
                                    size: 12
                                }
                            }
                        }
                    }
                }
            });
            
            charts.push(chart);
        }
    });
    
    return section;
}


// =============================================================================
// REAL-TIME DASHBOARD
// =============================================================================

async function initRealtimeDashboard(data) {
    console.log('Initializing real-time dashboard...');
    
    // Reset state
    analysisData = null;
    computedEvents = [];
    lastDisplayedFrame = -1;
    
    // Get video metadata from summary
    if (data.summary) {
        videoFps = data.summary.fps || 30;
        totalFrames = data.summary.total_frames || 0;
        
        // Use events from summary.json (SINGLE SOURCE OF TRUTH)
        // No recomputation needed - backend already computed these
        if (data.summary.events && Array.isArray(data.summary.events)) {
            computedEvents = data.summary.events;
            console.log('Events loaded from summary.json:', computedEvents.length);
        }
    }
    
    // Update total frames display
    document.getElementById('rtTotalFrames').textContent = `/ ${totalFrames}`;
    if (videoFps > 0 && totalFrames > 0) {
        const duration = totalFrames / videoFps;
        document.getElementById('rtDuration').textContent = `/ ${formatTimeRT(duration)}`;
    }
    
    // Load analysis.json for frame-by-frame data (used for current frame display only)
    try {
        const response = await fetch(`${API_BASE}/api/results/${data.output_dir}/analysis.json`);
        if (response.ok) {
            analysisData = await response.json();
            console.log('Analysis data loaded:', analysisData.analysis?.length, 'frames');
        } else {
            console.warn('Could not load analysis.json:', response.status);
        }
    } catch (error) {
        console.error('Error loading analysis data:', error);
    }
    
    // Build timeline visualization from pre-computed events
    buildActionTimeline();
    
    // Set up video timeupdate listener
    annotatedVideo.removeEventListener('timeupdate', onVideoTimeUpdate);
    annotatedVideo.addEventListener('timeupdate', onVideoTimeUpdate);
    
    // Also listen for seeking
    annotatedVideo.removeEventListener('seeked', onVideoTimeUpdate);
    annotatedVideo.addEventListener('seeked', onVideoTimeUpdate);
    
    // Initial update
    updateRealtimeDashboard(0);
}

function buildActionTimeline() {
    const timeline = document.getElementById('actionTimeline');
    if (!timeline) return;
    
    // Clear existing markers
    timeline.innerHTML = '';
    
    if (computedEvents.length === 0 || totalFrames === 0) return;
    
    // Add markers for each event
    for (const event of computedEvents) {
        const marker = document.createElement('div');
        marker.className = `timeline-marker ${event.type}`;
        const position = (event.frame / totalFrames) * 100;
        marker.style.left = `${position}%`;
        marker.title = `Frame ${event.frame}: ${event.text}`;
        timeline.appendChild(marker);
    }
}

function onVideoTimeUpdate() {
    if (!annotatedVideo) return;
    
    const currentTime = annotatedVideo.currentTime;
    const currentFrame = Math.floor(currentTime * videoFps);
    
    // Avoid redundant updates
    if (currentFrame === lastDisplayedFrame) return;
    lastDisplayedFrame = currentFrame;
    
    updateRealtimeDashboard(currentFrame);
}

function updateRealtimeDashboard(currentFrame) {
    // Update playback info
    updatePlaybackInfo(currentFrame);
    
    // Update action counts (cumulative up to current frame)
    updateActionCounts(currentFrame);
    
    // Update timeline playhead
    updateTimelinePlayhead(currentFrame);
    
    // Update fighter attributes (from current frame)
    updateFighterAttributes(currentFrame);
    
    // Update recent events
    updateRecentEvents(currentFrame);
}

function updatePlaybackInfo(currentFrame) {
    document.getElementById('rtFrame').textContent = currentFrame;
    
    if (videoFps > 0) {
        const currentTime = currentFrame / videoFps;
        document.getElementById('rtTime').textContent = formatTimeRT(currentTime);
    }
}

function updateActionCounts(currentFrame) {
    // Count events up to current frame
    const counts = { punch: 0, defense: 0, footwork: 0, clinch: 0 };
    
    for (const event of computedEvents) {
        if (event.frame <= currentFrame) {
            counts[event.type]++;
        }
    }
    
    // Update display with animation
    const punchEl = document.getElementById('rtPunchCount');
    const defenseEl = document.getElementById('rtDefenseCount');
    const footworkEl = document.getElementById('rtFootworkCount');
    const clinchEl = document.getElementById('rtClinchCount');
    
    updateCountWithHighlight(punchEl, counts.punch, 'punch');
    updateCountWithHighlight(defenseEl, counts.defense, 'defense');
    updateCountWithHighlight(footworkEl, counts.footwork, 'footwork');
    updateCountWithHighlight(clinchEl, counts.clinch, 'clinch');
}

function updateCountWithHighlight(element, newCount, type) {
    if (!element) return;
    
    const oldCount = parseInt(element.textContent) || 0;
    element.textContent = newCount;
    
    // Highlight if count increased
    if (newCount > oldCount) {
        const item = element.closest('.action-count-item');
        if (item) {
            item.classList.add('highlight');
            setTimeout(() => item.classList.remove('highlight'), 500);
        }
    }
}

function updateTimelinePlayhead(currentFrame) {
    const playhead = document.getElementById('timelinePlayhead');
    if (!playhead || totalFrames === 0) return;
    
    const position = (currentFrame / totalFrames) * 100;
    playhead.style.left = `${position}%`;
}

function updateFighterAttributes(currentFrame) {
    if (!analysisData || !analysisData.analysis) return;
    
    // Clamp frame index
    const frameIdx = Math.min(currentFrame, analysisData.analysis.length - 1);
    if (frameIdx < 0) return;
    
    const frameData = analysisData.analysis[frameIdx];
    const persons = frameData?.persons || [];
    
    // Find red and blue corner fighters
    let redFighter = null;
    let blueFighter = null;
    
    for (const person of persons) {
        const corner = person.corner || 'unknown';
        if (corner === 'red' && !redFighter) redFighter = person;
        if (corner === 'blue' && !blueFighter) blueFighter = person;
    }
    
    // Update red corner
    document.getElementById('rtRedGuard').textContent = redFighter?.guard || '—';
    document.getElementById('rtRedStance').textContent = redFighter?.stance || '—';
    document.getElementById('rtRedLeadHand').textContent = redFighter?.lead_hand || '—';
    
    // Update blue corner
    document.getElementById('rtBlueGuard').textContent = blueFighter?.guard || '—';
    document.getElementById('rtBlueStance').textContent = blueFighter?.stance || '—';
    document.getElementById('rtBlueLeadHand').textContent = blueFighter?.lead_hand || '—';
}

function updateRecentEvents(currentFrame) {
    const listEl = document.getElementById('recentEventsList');
    if (!listEl) return;
    
    // Get events up to current frame, take last 5
    const pastEvents = computedEvents.filter(e => e.frame <= currentFrame);
    const recentEvents = pastEvents.slice(-5).reverse();
    
    if (recentEvents.length === 0) {
        listEl.innerHTML = '<div class="no-events">Play video to see events</div>';
        return;
    }
    
    // Build event items
    listEl.innerHTML = recentEvents.map(event => {
        const time = videoFps > 0 ? formatTimeRT(event.frame / videoFps) : `F${event.frame}`;
        const fighterClass = event.fighter === 'red' || event.fighter === 'blue' ? event.fighter : '';
        
        return `
            <div class="event-item">
                <span class="event-time">${time}</span>
                <span class="event-dot ${event.type}"></span>
                <span class="event-text">
                    <span class="event-fighter ${fighterClass}">${event.fighter}</span>
                    ${event.action?.type || event.type}
                    ${event.action?.result ? '→ ' + event.action.result : ''}
                </span>
            </div>
        `;
    }).join('');
}

function formatTimeRT(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
