const state = {
    currentStep: 1,
    selectedCrypto: null,
    selectedCryptoName: null,
    daysAhead: 7,
    selectedModels: ['gru', 'lstm', 'hybrid_lstm', 'hybrid_gru'],
    predictions: null,
    futurePredictions: null,
    currentPrice: null,
    metrics: null,
    chart: null,
    modelSpecificChart: null,
    analysisTimer: null,
    analysisStartTime: null,
    abortController: null,
    trainingConfig: null,
    cryptoPrices: {},
    sortBy: 'name',
    showConfidence: false,
    showPercent: false
};

const advancedDefaults = {
    batchSize: 64,
    epochs: 200,
    transformerEpochs: 200,
    learningRate: 0.001,
    hiddenDim: 64,
    startDate: '2018-02-01',
    endDate: null,
    priceLagDays: 7,
    volumeLagDays: 1,
    fgiLagDays: 7,
    marketCapLagDays: 1,
    rsiWindow: 14,
    macdFast: 12,
    macdSlow: 26,
    macdSignal: 9,
    trainTestSplit: 0.8,
    numLayers: 2,
    transformerLearningRate: 0.0005,
};

const API_BASE = '';

document.addEventListener('DOMContentLoaded', initializeApp);

async function initializeApp() {
    await loadCryptocurrencies();
    setupEventListeners();
    setupHorizonOptions();
    setupModelOptions();
    setupAdvancedSettings();
    setupChartActions();
    initializeDateInputs();
}

function setupEventListeners() {
    // Step 1: Next button
    document.getElementById('next-step-1')?.addEventListener('click', () => goToStep(2));
    
    // Step 2: Navigation
    document.getElementById('prev-step-2')?.addEventListener('click', () => goToStep(1));
    document.getElementById('next-step-2')?.addEventListener('click', startAnalysis);
    document.getElementById('view-results')?.addEventListener('click', () => goToStep(4));
    
    // Step 3: Cancel training
    document.getElementById('cancel-training')?.addEventListener('click', cancelTraining);
    
    // Step 4: Actions
    document.getElementById('export-csv')?.addEventListener('click', exportResults);
    document.getElementById('new-prediction')?.addEventListener('click', resetApp);
    document.getElementById('back-to-config')?.addEventListener('click', () => goToStep(2));
    
    // Make completed steps clickable for navigation
    setupStepNavigation();
}

function setupHorizonOptions() {
    const select = document.getElementById('days-ahead-select');
    if (select) {
        // If select already has no options (edge-case), populate 1..30
        if (select.options.length === 0) {
            for (let d = 1; d <= 30; d++) {
                const opt = document.createElement('option');
                opt.value = String(d);
                opt.textContent = d === 1 ? '1 Day' : `${d} Days`;
                if (d === state.daysAhead) opt.selected = true;
                select.appendChild(opt);
            }
        }
        select.addEventListener('change', (e) => {
            state.daysAhead = parseInt(e.target.value);
        });
        // ensure initial value
        select.value = String(state.daysAhead || 7);
    } else {
        // fallback for older radio-based markup
        const options = document.querySelectorAll('input[name="days-ahead"]');
        options.forEach(option => {
            option.addEventListener('change', (e) => {
                state.daysAhead = parseInt(e.target.value);
            });
        });
    }
}

function setupModelOptions() {
    const options = document.querySelectorAll('input[name="models"]');
    options.forEach(option => {
        option.addEventListener('change', updateSelectedModels);
    });
}

function updateSelectedModels() {
    const checkboxes = document.querySelectorAll('input[name="models"]:checked');
    state.selectedModels = Array.from(checkboxes).map(cb => cb.value);
    
    // Disable start analysis button if no models selected
    const startBtn = document.getElementById('next-step-2');
    if (startBtn) {
        startBtn.disabled = state.selectedModels.length === 0;
        if (state.selectedModels.length === 0) {
            startBtn.textContent = 'Select at least one model';
        } else {
            startBtn.innerHTML = 'Start Analysis <span class="btn-arrow">→</span>';
        }
    }
}

/**
 * Setup advanced settings toggle
 */
function setupAdvancedSettings() {
    const toggle = document.getElementById('advanced-toggle');
    const settings = document.getElementById('advanced-settings');
    
    if (toggle && settings) {
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('open');
            settings.classList.toggle('hidden');
        });
    }
}

/**
 * Initialize date input defaults
 */
function initializeDateInputs() {
    const endDateInput = document.getElementById('end-date');
    if (endDateInput) {
        // Set end date to today
        const today = new Date().toISOString().split('T')[0];
        endDateInput.value = today;
        advancedDefaults.endDate = today;
    }
}

/**
 * Setup chart action buttons
 */
function setupChartActions() {
    const percentBtn = document.getElementById('toggle-percent');
    const confBtn = document.getElementById('toggle-confidence');

    if (percentBtn) {
        percentBtn.addEventListener('click', () => {
            // If already in percent mode, turn it off (back to price, no CI)
            if (state.showPercent) {
                state.showPercent = false;
                state.showConfidence = false;
                percentBtn.classList.remove('active');
                confBtn?.classList.remove('active');
            } else {
                // Switch to percent mode, turn off CI
                state.showPercent = true;
                state.showConfidence = false;
                percentBtn.classList.add('active');
                confBtn?.classList.remove('active');
            }
            if (state.chart) renderChart();
        });
    }

    if (confBtn) {
        confBtn.addEventListener('click', () => {
            // If already showing confidence, turn it off
            if (state.showConfidence) {
                state.showConfidence = false;
                confBtn.classList.remove('active');
            } else {
                // Switch to confidence bands, turn off percent mode
                state.showConfidence = true;
                state.showPercent = false;
                confBtn.classList.add('active');
                percentBtn?.classList.remove('active');
            }
            if (state.chart) renderChart();
        });
    }

    // Model-specific tabs
    document.querySelectorAll('.model-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            renderModelSpecificChart(e.target.dataset.model);
        });
    });
}

/**
 * Setup clickable step navigation for completed steps
 */
function setupStepNavigation() {
    document.querySelectorAll('.progress-steps .step').forEach(step => {
        step.style.cursor = 'pointer';
        step.addEventListener('click', () => {
            const stepNum = parseInt(step.dataset.step);
            // Only allow navigation to completed steps or current step - 1
            // Don't allow going back during analysis (step 3)
            if (state.currentStep === 3) return; // Can't navigate during analysis
            
            if (stepNum < state.currentStep || step.classList.contains('completed')) {
                // If going back to step 1, need to enable proper state
                if (stepNum === 1) {
                    // Keep the crypto selected
                    document.getElementById('next-step-1').disabled = !state.selectedCrypto;
                }
                goToStep(stepNum);
            }
        });
    });
}

// ============================================
// Step Navigation
// ============================================
function goToStep(stepNumber) {
    // Don't allow navigation during analysis unless it's step 4 (completion) or going back on error
    const analysisIcon = document.getElementById('analysis-icon')?.textContent;
    const isAnalysisRunning = state.currentStep === 3 && analysisIcon !== '❌' && analysisIcon !== '✅';
    if (isAnalysisRunning && stepNumber !== 4) {
        return;
    }
    
    const previousStep = state.currentStep;
    const direction = stepNumber > previousStep ? 'right' : 'left';
    
    // Update step indicators based on where we're going
    document.querySelectorAll('.progress-steps .step').forEach(step => {
        const num = parseInt(step.dataset.step);
        step.classList.remove('active', 'completed');
        
        // Mark steps as completed if:
        // 1. They are before the current step number, OR
        // 2. We have results and we're navigating back (keep step 3 & 4 as completed)
        if (num < stepNumber) {
            step.classList.add('completed');
        } else if (num === stepNumber) {
            step.classList.add('active');
        } else if (state.predictions && num <= 4 && stepNumber < 4) {
            // Keep higher steps marked as completed if we have results and are going back
            step.classList.add('completed');
        }
    });
    
    // Animate step transition
    const currentContent = document.getElementById(`step-${previousStep}`);
    const nextContent = document.getElementById(`step-${stepNumber}`);
    
    if (currentContent && nextContent && previousStep !== stepNumber) {
        // Hide current with animation
        currentContent.classList.remove('active', 'slide-in-right', 'slide-in-left');
        currentContent.classList.add(direction === 'right' ? 'slide-out-left' : 'slide-out-right');
        
        setTimeout(() => {
            currentContent.classList.remove('slide-out-left', 'slide-out-right');
            currentContent.classList.remove('active');
            
            // Show next with animation
            nextContent.classList.add('active', direction === 'right' ? 'slide-in-right' : 'slide-in-left');
        }, 300);
    } else if (nextContent) {
        document.querySelectorAll('.step-content').forEach(content => {
            content.classList.remove('active');
        });
        nextContent.classList.add('active');
    }
    
    state.currentStep = stepNumber;
    
    // Step-specific updates
    if (stepNumber === 2) {
        document.getElementById('selected-crypto-name').textContent = 
            `${state.selectedCrypto} - ${state.selectedCryptoName}`;
        
        // Show "View Results" button if we already have predictions for this crypto
        const viewResultsBtn = document.getElementById('view-results');
        if (viewResultsBtn) {
            if (state.predictions && state.futurePredictions) {
                viewResultsBtn.classList.remove('hidden');
            } else {
                viewResultsBtn.classList.add('hidden');
            }
        }
    }
}

// ============================================
// API Functions
// ============================================
async function apiCall(endpoint, options = {}) {
    try {
        // Add abort signal if controller exists
        const fetchOptions = {
            headers: { 'Content-Type': 'application/json' },
            ...options
        };
        
        if (state.abortController) {
            fetchOptions.signal = state.abortController.signal;
        }
        
        const response = await fetch(`${API_BASE}${endpoint}`, fetchOptions);
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    } catch (error) {
        // Don't log abort errors as they're intentional
        if (error.name === 'AbortError') {
            throw error;
        }
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// ============================================
// Load Cryptocurrencies
// ============================================
async function loadCryptocurrencies() {
    try {
        const data = await apiCall('/api/cryptocurrencies');
        renderCryptoGrid(data.cryptocurrencies);
    } catch (error) {
        console.error('Failed to load cryptocurrencies:', error);
        showToast('Failed to load cryptocurrencies', 'error');
    }
}

function renderCryptoGrid(cryptos) {
    const grid = document.getElementById('crypto-grid');
    if (!grid) return;

    // Remove skeletons
    grid.innerHTML = '';

    // Render actual crypto cards
    cryptos.forEach(crypto => {
        const card = document.createElement('div');
        card.className = 'crypto-card';
        card.dataset.symbol = crypto.symbol;
        card.dataset.name = crypto.name;

        card.innerHTML = `
            <div class="crypto-symbol">${crypto.symbol}</div>
            <div class="crypto-name">${crypto.name}</div>
        `;

        card.addEventListener('click', () => selectCrypto(card));
        grid.appendChild(card);
    });
}

function selectCrypto(card) {
    // Remove selection from all cards
    document.querySelectorAll('.crypto-card').forEach(c => c.classList.remove('selected'));
    
    // Select this card
    card.classList.add('selected');
    
    state.selectedCrypto = card.dataset.symbol;
    state.selectedCryptoName = card.dataset.name;
    
    // Enable next button
    document.getElementById('next-step-1').disabled = false;
}

// ============================================
// Analysis Flow
// ============================================
async function startAnalysis() {
    // Validate at least one model is selected
    if (state.selectedModels.length === 0) {
        showToast('Please select at least one AI model', 'error');
        return;
    }
    
    goToStep(3);
    
    // Reset stages first (this resets the title HTML)
    resetStages();
    
    // Create abort controller for cancellation support
    state.abortController = new AbortController();
    
    // Update UI with selected crypto (after reset so it's not overwritten)
    document.getElementById('analyzing-crypto').textContent = state.selectedCrypto;
    
    // Start timer
    startTimer();
    
    try {
        // Stage 1: Fetch Data
        await runStage('fetch', fetchData);
        
        // Stage 2: Train Models
        await runStage('train', trainModels);
        
        // Stage 3: Generate Predictions
        await runStage('predict', generatePredictions);
        
        // Complete
        state.abortController = null;  // Clear abort controller on success
        completeAnalysis();
        
    } catch (error) {
        // Handle abort errors silently (user cancelled)
        if (error.name === 'AbortError') {
            return;
        }
        
        console.error('Analysis failed:', error);
        showToast(`Analysis failed: ${error.message}`, 'error');
        stopTimer();
        state.abortController = null;  // Clear abort controller on error
        
        // Reset all stages to remove active/completed states
        ['fetch', 'train', 'predict'].forEach(stage => {
            const el = document.getElementById(`stage-${stage}`);
            el.classList.remove('active', 'completed');
            el.querySelector('.stage-bar').style.width = '0%';
        });
        
        // Update UI to show error with retry option
        document.getElementById('analysis-icon').textContent = '❌';
        document.getElementById('analysis-icon').classList.remove('spinning');
        document.getElementById('analysis-title').textContent = 'Analysis Failed';
        document.getElementById('analysis-subtitle').innerHTML = 
            `${error.message}<br><br>
            <button class="btn btn-secondary" onclick="goToStep(2)" style="margin-right: 0.5rem;">← Back to Settings</button>
            <button class="btn btn-primary" onclick="startAnalysis()">Retry Analysis</button>`;
    }
}

/**
 * Cancel ongoing training
 * Note: Training runs synchronously on the server, so we can only
 * abort the request and return to config. The server-side training
 * will complete but results will be ignored.
 */
function cancelTraining() {
    // Show confirmation dialog
    if (!confirm('Are you sure you want to cancel training? This will stop the current analysis.')) {
        return;
    }
    
    // Abort any pending fetch requests
    if (state.abortController) {
        state.abortController.abort();
        state.abortController = null;
    }
    
    showToast('Training cancelled', 'info');
    stopTimer();
    
    // Force navigation by updating state first
    state.currentStep = 1;
    
    // Hide step 3, show step 1
    document.getElementById('step-3').classList.remove('active');
    document.getElementById('step-1').classList.add('active');
    
    // Update step indicators
    document.querySelectorAll('.progress-steps .step').forEach(step => {
        const num = parseInt(step.dataset.step);
        step.classList.remove('active', 'completed');
        if (num === 1) {
            step.classList.add('active');
        }
    });
    
    // Reset stages for next analysis
    resetStages();
}

function resetStages() {
    ['fetch', 'train', 'predict'].forEach(stage => {
        const el = document.getElementById(`stage-${stage}`);
        el.classList.remove('active', 'completed');
        el.querySelector('.stage-bar').style.width = '0%';
    });
    
    document.getElementById('training-models').innerHTML = '';
    document.getElementById('analysis-icon').textContent = '⚙️';
    document.getElementById('analysis-icon').classList.add('spinning');
    document.getElementById('analysis-icon').classList.remove('complete');
    document.getElementById('analysis-title').innerHTML = 'Analyzing <span id="analyzing-crypto">--</span>';
    document.getElementById('analysis-subtitle').textContent = 'This may take a few minutes...';
}

async function runStage(stageName, stageFunction) {
    const stageEl = document.getElementById(`stage-${stageName}`);
    stageEl.classList.add('active');
    
    const startTime = Date.now();
    await stageFunction();
    
    // Ensure minimum visibility of stage progress (at least 500ms)
    const elapsed = Date.now() - startTime;
    if (elapsed < 500) {
        await new Promise(resolve => setTimeout(resolve, 500 - elapsed));
    }
    
    stageEl.classList.remove('active');
    stageEl.classList.add('completed');
}

function startTimer() {
    state.analysisStartTime = Date.now();
    state.analysisTimer = setInterval(updateTimer, 1000);
}

function stopTimer() {
    if (state.analysisTimer) {
        clearInterval(state.analysisTimer);
        state.analysisTimer = null;
    }
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - state.analysisStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    document.getElementById('elapsed-time').textContent = `${minutes}:${seconds}`;
}

// ============================================
// Stage Functions
// ============================================
async function fetchData() {
    await apiCall('/api/data/fetch', {
        method: 'POST',
        body: JSON.stringify({ 
            symbol: state.selectedCrypto,
            force_refresh: false 
        })
    });
}

async function trainModels() {
    const modelsDiv = document.getElementById('training-models');
    modelsDiv.innerHTML = state.selectedModels.map(m => 
        `<div class="model-status" id="model-${m}">Training ${formatModelName(m)}...</div>`
    ).join('');
    
    // Get advanced settings
    const batchSize = parseInt(document.getElementById('batch-size')?.value) || advancedDefaults.batchSize;
    const epochs = parseInt(document.getElementById('epochs')?.value) || advancedDefaults.epochs;
    const learningRate = parseFloat(document.getElementById('learning-rate')?.value) || advancedDefaults.learningRate;
    const hiddenDim = parseInt(document.getElementById('hidden-dim')?.value) || advancedDefaults.hiddenDim;
    const startDate = document.getElementById('start-date')?.value || advancedDefaults.startDate;
    const endDate = document.getElementById('end-date')?.value || advancedDefaults.endDate;

    // New advanced settings
    const priceLagDays = parseInt(document.getElementById('price-lag-days')?.value) || advancedDefaults.priceLagDays;
    const volumeLagDays = parseInt(document.getElementById('volume-lag-days')?.value) || advancedDefaults.volumeLagDays;
    const fgiLagDays = parseInt(document.getElementById('fgi-lag-days')?.value) || advancedDefaults.fgiLagDays;
    const marketCapLagDays = parseInt(document.getElementById('marketcap-lag-days')?.value) || advancedDefaults.marketCapLagDays;
    const rsiWindow = parseInt(document.getElementById('rsi-window')?.value) || advancedDefaults.rsiWindow;
    const macdFast = parseInt(document.getElementById('macd-fast')?.value) || advancedDefaults.macdFast;
    const macdSlow = parseInt(document.getElementById('macd-slow')?.value) || advancedDefaults.macdSlow;
    const macdSignal = parseInt(document.getElementById('macd-signal')?.value) || advancedDefaults.macdSignal;
    const trainTestSplit = parseFloat(document.getElementById('train-test-split')?.value) || advancedDefaults.trainTestSplit;
    const numLayers = parseInt(document.getElementById('num-layers')?.value) || advancedDefaults.numLayers;
    const transformerLearningRate = parseFloat(document.getElementById('transformer-learning-rate')?.value) || advancedDefaults.transformerLearningRate;
    const transformerEpochs = parseInt(document.getElementById('transformer-epochs')?.value) || advancedDefaults.transformerEpochs;
    
    // Store training config for results display
    state.trainingConfig = {
        epochs: epochs,
        batchSize: batchSize,
        learningRate: learningRate,
        hiddenDim: hiddenDim,
        priceLagDays, volumeLagDays, fgiLagDays, marketCapLagDays,
        rsiWindow, macdFast, macdSlow, macdSignal,
        trainTestSplit, numLayers, transformerLearningRate, transformerEpochs
    };
    
    await apiCall('/api/train', {
        method: 'POST',
        body: JSON.stringify({ 
            symbol: state.selectedCrypto,
            model_types: state.selectedModels,
            batch_size: batchSize,
            epochs: epochs,
            learning_rate: learningRate,
            hidden_dim: hiddenDim,
            start_date: startDate,
            end_date: endDate,
            // feature-engineering
            price_lag_days: priceLagDays,
            volume_lag_days: volumeLagDays,
            fgi_lag_days: fgiLagDays,
            market_cap_lag_days: marketCapLagDays,
            rsi_window: rsiWindow,
            macd_fast: macdFast,
            macd_slow: macdSlow,
            macd_signal: macdSignal,
            // other
            train_test_split: trainTestSplit,
            num_layers: numLayers,
            transformer_learning_rate: transformerLearningRate,
            transformer_epochs: transformerEpochs
        })
    });
    
    modelsDiv.innerHTML = state.selectedModels.map(m => 
        `<div class="model-status" id="model-${m}">✓ ${formatModelName(m)} trained</div>`
    ).join('');
}

async function generatePredictions() {
    // Get current predictions
    state.predictions = await apiCall('/api/predict', {
        method: 'POST',
        body: JSON.stringify({ 
            symbol: state.selectedCrypto,
            include_confidence: true
        })
    });
    
    // Get future predictions
    const futureData = await apiCall('/api/predict/future', {
        method: 'POST',
        body: JSON.stringify({ 
            symbol: state.selectedCrypto,
            days_ahead: state.daysAhead
        })
    });
    state.futurePredictions = futureData.predictions;
    state.currentPrice = futureData.current_price;  // Store current price from API
    
    // Get metrics
    const metricsData = await apiCall('/api/evaluate', {
        method: 'POST',
        body: JSON.stringify({ symbol: state.selectedCrypto })
    });
    state.metrics = metricsData.metrics;
}

function completeAnalysis() {
    // Store training time before stopping timer
    const elapsedTime = document.getElementById('elapsed-time')?.textContent || '00:00';
    state.trainingTime = elapsedTime;
    
    stopTimer();
    
    // Update analysis panel
    document.getElementById('analysis-icon').textContent = '✅';
    document.getElementById('analysis-icon').classList.remove('spinning');
    document.getElementById('analysis-title').textContent = 'Analysis Complete!';
    document.getElementById('analysis-subtitle').textContent = 'Preparing results...';
    
    // Short delay then show results
    setTimeout(() => {
        renderResults();
        goToStep(4);
    }, 1000);
}

// ============================================
// Results Rendering
// ============================================
function renderResults() {
    // Header
    document.getElementById('results-crypto').textContent = 
        `${state.selectedCrypto} (${state.selectedCryptoName})`;
    document.getElementById('results-date').textContent = 
        new Date().toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
    
    // Training Summary
    renderTrainingSummary();
    
    // Quick stats
    renderQuickStats();
    
    // Chart
    renderChart();
    
    // Model-specific chart (first selected model)
    if (state.selectedModels.length > 0) {
        renderModelSpecificChart(state.selectedModels[0]);
        
        // Update active tab
        document.querySelectorAll('.model-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.model === state.selectedModels[0]);
            // Hide tabs for non-selected models
            if (!state.selectedModels.includes(tab.dataset.model)) {
                tab.style.display = 'none';
            } else {
                tab.style.display = '';
            }
        });
    }
    
    // Accuracy scoreboard
    renderAccuracyScoreboard();
    
    // Predictions table
    renderPredictionsTable();
    
    // Metrics
    renderMetrics();
}

/**
 * Render training summary
 */
function renderTrainingSummary() {
    if (!state.trainingConfig) return;
    
    document.getElementById('summary-epochs').textContent = state.trainingConfig.epochs;
    document.getElementById('summary-batch-size').textContent = state.trainingConfig.batchSize;
    document.getElementById('summary-learning-rate').textContent = state.trainingConfig.learningRate;
    document.getElementById('summary-hidden-dim').textContent = state.trainingConfig.hiddenDim;
    document.getElementById('summary-models').textContent = state.selectedModels.length;
    document.getElementById('summary-time').textContent = state.trainingTime || '--';
}

/**
 * Render accuracy scoreboard
 */
function renderAccuracyScoreboard() {
    const container = document.getElementById('accuracy-scoreboard');
    if (!container || !state.metrics) return;
    
    // Calculate accuracy scores (inverse of MAPE, capped at 100%)
    const scores = {};
    let bestModel = null;
    let bestScore = 0;
    
    Object.entries(state.metrics)
        .filter(([model]) => state.selectedModels.includes(model))
        .forEach(([model, values]) => {
            const accuracy = Math.max(0, 100 - values.MAPE);
            scores[model] = accuracy;
            if (accuracy > bestScore) {
                bestScore = accuracy;
                bestModel = model;
            }
        });
    
    // Sort by score
    const sortedModels = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .map(([model], index) => ({ model, score: scores[model], rank: index + 1 }));
    
    container.innerHTML = sortedModels.map(({ model, score, rank }) => `
        <div class="accuracy-card ${model === bestModel ? 'winner' : ''}">
            <div class="model-name">${formatModelName(model)}</div>
            <div class="accuracy-value">${score.toFixed(1)}%</div>
            <div class="accuracy-label">Accuracy</div>
            <div class="accuracy-rank">${rank === 1 ? '🏆 Best' : `#${rank}`}</div>
        </div>
    `).join('');
}

/**
 * Render model-specific chart
 */
function renderModelSpecificChart(model) {
    const canvas = document.getElementById('model-specific-chart');
    if (!canvas || !state.predictions) return;
    
    const ctx = canvas.getContext('2d');
    
    if (state.modelSpecificChart) {
        state.modelSpecificChart.destroy();
    }
    
    const colors = {
        gru: '#8B5CF6',
        lstm: '#EC4899',
        hybrid_lstm: '#10B981',
        hybrid_gru: '#F59E0B'
    };
    
    const modelColor = colors[model] || '#667eea';
    
    // Prepare data
    const dates = state.predictions.dates.map(d => new Date(d));
    const actual = state.predictions.actual;
    const predicted = state.predictions.predictions[model] || [];
    
    // Calculate error
    const error = actual.map((a, i) => Math.abs(a - predicted[i]));
    
    state.modelSpecificChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Actual',
                    data: dates.map((d, i) => ({ x: d, y: actual[i] })),
                    borderColor: '#374151',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: formatModelName(model),
                    data: dates.map((d, i) => ({ x: d, y: predicted[i] })),
                    borderColor: modelColor,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    title: { display: false }
                },
                y: {
                    title: { display: true, text: 'Price (USD)' }
                }
            },
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${formatPrice(context.parsed.y)}`;
                        }
                    }
                }
            }
        }
    });
}

function renderQuickStats() {
    if (!state.futurePredictions || state.futurePredictions.length === 0) return;
    
    // Find forecast data
    const forecasts = state.futurePredictions.filter(p => p.type === 'forecast');
    
    // Current price - use the value from API (last known price from processed_data)
    const currentPrice = state.currentPrice || 0;
    
    // Predicted price (last forecast ensemble average - only from selected models)
    const lastForecast = forecasts[forecasts.length - 1];
    const predictedPrice = lastForecast ? 
        calculateEnsemblePrice(lastForecast) : 
        currentPrice;
    
    // Price change - avoid NaN
    const priceChange = currentPrice > 0 ? ((predictedPrice - currentPrice) / currentPrice) * 100 : 0;
    
    // Best model (lowest RMSE)
    let bestModel = 'gru';
    let lowestRMSE = Infinity;
    if (state.metrics) {
        Object.entries(state.metrics).forEach(([model, values]) => {
            if (values.RMSE < lowestRMSE) {
                lowestRMSE = values.RMSE;
                bestModel = model;
            }
        });
    }
    
    // Update UI
    document.getElementById('current-price').textContent = formatPrice(currentPrice);
    document.getElementById('predicted-price').textContent = formatPrice(predictedPrice);
    
    const changeEl = document.getElementById('price-change');
    changeEl.textContent = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`;
    changeEl.className = `stat-number ${priceChange >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('trend-icon').textContent = priceChange >= 0 ? '📈' : '📉';
    document.getElementById('best-model').textContent = formatModelName(bestModel);
}

/**
 * Setup chart filter toggle buttons
 */
function setupChartToggles() {
    const toggleContainer = document.getElementById('chart-toggles');
    if (!toggleContainer || !state.chart) return;
    
    // Map series names to dataset labels
    const seriesMap = {
        'actual': 'Actual',
        'gru': 'Bi-GRU',
        'lstm': 'Bi-LSTM',
        'hybrid_lstm': 'Hybrid LSTM',
        'hybrid_gru': 'Hybrid GRU'
    };
    
    // Update toggle visibility based on selected models
    toggleContainer.querySelectorAll('.filter-toggle').forEach(toggle => {
        const series = toggle.dataset.series;
        if (series !== 'actual' && !state.selectedModels.includes(series)) {
            toggle.style.display = 'none';
        } else {
            toggle.style.display = '';
        }
    });
    
    // Add click handlers
    toggleContainer.querySelectorAll('.filter-toggle').forEach(toggle => {
        // Remove any existing listeners by cloning
        const newToggle = toggle.cloneNode(true);
        toggle.parentNode.replaceChild(newToggle, toggle);
        
        newToggle.addEventListener('click', () => {
            const series = newToggle.dataset.series;
            const isActive = newToggle.classList.contains('active');
            
            // Toggle the active state
            newToggle.classList.toggle('active');
            
            // Find matching datasets and toggle visibility
            const label = seriesMap[series];
            state.chart.data.datasets.forEach((dataset, index) => {
                // Match both regular and forecast versions
                if (dataset.label === label || dataset.label === `${label} (Forecast)`) {
                    state.chart.setDatasetVisibility(index, !isActive);
                }
            });
            
            state.chart.update();
        });
    });
}

/**
 * Setup date range toggle buttons for chart zoom
 */
function setupRangeToggles() {
    const rangeContainer = document.getElementById('range-toggles');
    if (!rangeContainer || !state.chart) return;
    
    rangeContainer.querySelectorAll('.range-toggle').forEach(toggle => {
        // Remove any existing listeners by cloning
        const newToggle = toggle.cloneNode(true);
        toggle.parentNode.replaceChild(newToggle, toggle);
        
        newToggle.addEventListener('click', () => {
            const range = newToggle.dataset.range;
            
            // Update active state
            rangeContainer.querySelectorAll('.range-toggle').forEach(t => t.classList.remove('active'));
            newToggle.classList.add('active');
            
            // Calculate date range
            const dates = state.predictions.dates.map(d => new Date(d));
            const maxDate = new Date(Math.max(...dates));
            let minDate;
            
            switch (range) {
                case '1m':
                    minDate = new Date(maxDate);
                    minDate.setMonth(minDate.getMonth() - 1);
                    break;
                case '3m':
                    minDate = new Date(maxDate);
                    minDate.setMonth(minDate.getMonth() - 3);
                    break;
                case '6m':
                    minDate = new Date(maxDate);
                    minDate.setMonth(minDate.getMonth() - 6);
                    break;
                case 'all':
                default:
                    minDate = new Date(Math.min(...dates));
                    break;
            }
            
            // Add some padding for future predictions
            const futureMaxDate = new Date(maxDate);
            futureMaxDate.setDate(futureMaxDate.getDate() + (state.daysAhead || 7) + 2);
            
            // Update chart x-axis
            state.chart.options.scales.x.min = minDate.getTime();
            state.chart.options.scales.x.max = futureMaxDate.getTime();
            
            // Adjust time unit based on range
            if (range === '1m') {
                state.chart.options.scales.x.time.unit = 'day';
            } else if (range === '3m') {
                state.chart.options.scales.x.time.unit = 'week';
            } else {
                state.chart.options.scales.x.time.unit = 'month';
            }
            
            state.chart.update();
        });
    });
}

function renderChart() {
    const canvas = document.getElementById('price-chart');
    if (!canvas || !state.predictions) return;
    
    const ctx = canvas.getContext('2d');
    
    if (state.chart) {
        state.chart.destroy();
    }
    
    const colors = {
        actual: { line: '#374151', fill: 'rgba(55, 65, 81, 0.1)' },
        gru: { line: '#8B5CF6', fill: 'rgba(139, 92, 246, 0.1)' },
        lstm: { line: '#EC4899', fill: 'rgba(236, 72, 153, 0.1)' },
        hybrid_lstm: { line: '#10B981', fill: 'rgba(16, 185, 129, 0.1)' },
        hybrid_gru: { line: '#F59E0B', fill: 'rgba(245, 158, 11, 0.1)' }
    };
    
    // Base price for percentage calculation
    const basePrice = state.predictions.actual[0];
    
    // Transform data to percentage if needed
    const transformValue = (value) => {
        if (state.showPercent && basePrice > 0) {
            return ((value - basePrice) / basePrice) * 100;
        }
        return value;
    };
    
    const datasets = [
        {
            label: 'Actual',
            data: state.predictions.dates.map((date, i) => ({ 
                x: new Date(date), 
                y: transformValue(state.predictions.actual[i])
            })),
            borderColor: colors.actual.line,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
        }
    ];
    
    // Add model predictions
    Object.entries(state.predictions.predictions).forEach(([model, values]) => {
        if (state.selectedModels.includes(model)) {
            const color = colors[model] || { line: '#666', fill: 'rgba(102,102,102,0.1)' };
            datasets.push({
                label: formatModelName(model),
                data: state.predictions.dates.map((date, i) => ({ 
                    x: new Date(date), 
                    y: transformValue(values[i])
                })),
                borderColor: color.line,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            });
            
            // Add confidence intervals if enabled
            if (state.showConfidence && state.predictions.confidence && state.predictions.confidence[model]) {
                const conf = state.predictions.confidence[model];
                datasets.push({
                    label: `${formatModelName(model)} CI`,
                    data: state.predictions.dates.map((date, i) => ({
                        x: new Date(date),
                        y: transformValue(conf.upper[i])
                    })),
                    borderColor: 'transparent',
                    backgroundColor: color.fill,
                    borderWidth: 0,
                    pointRadius: 0,
                    fill: '+1',
                    tension: 0.1
                });
                datasets.push({
                    label: `${formatModelName(model)} CI Lower`,
                    data: state.predictions.dates.map((date, i) => ({
                        x: new Date(date),
                        y: transformValue(conf.lower[i])
                    })),
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    borderWidth: 0,
                    pointRadius: 0,
                    tension: 0.1
                });
            }
        }
    });
    
    // Add future predictions
    if (state.futurePredictions) {
        const forecasts = state.futurePredictions.filter(p => p.type === 'forecast');
        if (forecasts.length > 0) {
            const lastActualDate = new Date(state.predictions.dates[state.predictions.dates.length - 1]);
            const lastActualValue = state.predictions.actual[state.predictions.actual.length - 1];
            
            Object.keys(colors).forEach(model => {
                if (model !== 'actual' && state.selectedModels.includes(model)) {
                    const forecastData = [
                        { x: lastActualDate, y: transformValue(lastActualValue) },
                        ...forecasts.map(f => ({ x: new Date(f.date), y: transformValue(f[model]) }))
                    ];
                    
                    datasets.push({
                        label: `${formatModelName(model)} (Forecast)`,
                        data: forecastData,
                        borderColor: colors[model].line,
                        backgroundColor: colors[model].fill,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 4,
                        pointBackgroundColor: colors[model].line,
                        fill: true,
                        tension: 0.1
                    });
                }
            });
        }
    }
    
    state.chart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    type: 'time',
                    time: { 
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d',
                            week: 'MMM d',
                            month: 'MMM yyyy'
                        }
                    },
                    title: { display: true, text: 'Date' }
                },
                y: {
                    title: { 
                        display: true, 
                        text: state.showPercent ? 'Change (%)' : `${state.selectedCrypto} Price (USD)` 
                    }
                }
            },
            plugins: {
                legend: { 
                    display: false  // Hide built-in legend, using custom filter toggles
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (state.showPercent) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
                            }
                            return `${context.dataset.label}: ${formatPrice(context.parsed.y)}`;
                        }
                    },
                    filter: function(tooltipItem) {
                        // Only show tooltip for visible datasets and non-CI datasets
                        const hidden = tooltipItem.chart.getDatasetMeta(tooltipItem.datasetIndex).hidden;
                        const isCI = tooltipItem.dataset.label && tooltipItem.dataset.label.includes('CI');
                        return !hidden && !isCI;
                    }
                },
                annotation: {}
            }
        }
    });
    
    // Setup chart filter toggles and range controls
    setupChartToggles();
    setupRangeToggles();
    
    // All datasets visible by default (toggles already have 'active' class)
    state.chart.update();
}

function renderPredictionsTable() {
    const tbody = document.getElementById('predictions-tbody');
    if (!tbody || !state.futurePredictions) return;
    
    // Get current price from state (last known actual price)
    const currentPrice = state.currentPrice || 0;
    
    // Build a map of dates to actual prices from predictions data
    // Normalize dates to YYYY-MM-DD format for matching
    const actualPriceMap = {};
    if (state.predictions && state.predictions.dates && state.predictions.actual) {
        state.predictions.dates.forEach((date, i) => {
            // Extract just the date part (YYYY-MM-DD) for matching
            const dateKey = date.split('T')[0];
            actualPriceMap[dateKey] = state.predictions.actual[i];
        });
    }
    
    tbody.innerHTML = state.futurePredictions.map((row, idx) => {
        const ensemble = calculateEnsemblePrice(row);
        const trend = row.type === 'forecast' ? 
            (ensemble > currentPrice ? 
                '<span class="trend-up">↑ Bullish</span>' : 
                '<span class="trend-down">↓ Bearish</span>') : 
            '-';
        
        // Get actual price for this date (only available for historical rows)
        // Normalize the row date to YYYY-MM-DD format for matching
        const rowDateKey = row.date ? row.date.split('T')[0] : '';
        const actualPrice = actualPriceMap[rowDateKey];
        const actualCell = row.type === 'historical' && actualPrice != null ? 
            formatPrice(actualPrice) : '-';
        
        // Only show columns for selected models
        const gruCell = state.selectedModels.includes('gru') ? formatPrice(row.gru) : '-';
        const lstmCell = state.selectedModels.includes('lstm') ? formatPrice(row.lstm) : '-';
        const hybridLstmCell = state.selectedModels.includes('hybrid_lstm') ? formatPrice(row.hybrid_lstm) : '-';
        const hybridGruCell = state.selectedModels.includes('hybrid_gru') ? formatPrice(row.hybrid_gru) : '-';
        
        return `
            <tr class="${row.type === 'forecast' ? 'forecast-row' : ''}">
                <td>${formatDate(row.date)}</td>
                <td><span class="type-badge ${row.type}">${row.type}</span></td>
                <td class="price-cell">${actualCell}</td>
                <td class="price-cell">${gruCell}</td>
                <td class="price-cell">${lstmCell}</td>
                <td class="price-cell">${hybridLstmCell}</td>
                <td class="price-cell">${hybridGruCell}</td>
                <td class="price-cell"><strong>${formatPrice(ensemble)}</strong></td>
                <td>${trend}</td>
            </tr>
        `;
    }).join('');
}

function renderMetrics() {
    const container = document.getElementById('metrics-container');
    if (!container || !state.metrics) return;
    
    // Find best values
    const bestValues = {
        RMSE: Math.min(...Object.values(state.metrics).map(m => m.RMSE)),
        MAE: Math.min(...Object.values(state.metrics).map(m => m.MAE)),
        MAPE: Math.min(...Object.values(state.metrics).map(m => m.MAPE))
    };
    
    container.innerHTML = Object.entries(state.metrics)
        .filter(([model]) => state.selectedModels.includes(model))
        .map(([model, values]) => `
            <div class="metric-card">
                <h4>${formatModelName(model)}</h4>
                <div class="metric-values">
                    <div class="metric-item ${values.RMSE === bestValues.RMSE ? 'best' : ''}">
                        <span class="metric-label">RMSE</span>
                        <span class="metric-value">${formatNumber(values.RMSE)}</span>
                    </div>
                    <div class="metric-item ${values.MAE === bestValues.MAE ? 'best' : ''}">
                        <span class="metric-label">MAE</span>
                        <span class="metric-value">${formatNumber(values.MAE)}</span>
                    </div>
                    <div class="metric-item ${values.MAPE === bestValues.MAPE ? 'best' : ''}">
                        <span class="metric-label">MAPE</span>
                        <span class="metric-value">${values.MAPE.toFixed(4)}%</span>
                    </div>
                </div>
            </div>
        `).join('');
}

// ============================================
// Export Results
// ============================================
async function exportResults() {
    if (!state.futurePredictions) return;
    
    try {
        const csv = convertToCSV(state.futurePredictions);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().slice(0, 10);
        
        link.href = URL.createObjectURL(blob);
        link.download = `${state.selectedCrypto}_predictions_${timestamp}.csv`;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showToast('CSV downloaded successfully!');
    } catch (error) {
        showToast(`Export failed: ${error.message}`, 'error');
    }
}

/**
 * Convert predictions to CSV format (similar to notebook output)
 * Format: date, GRU, LSTM, Hybrid_LSTM, Hybrid_GRU, GRU_lower, GRU_upper, LSTM_lower, LSTM_upper, etc.
 */
function convertToCSV(predictions) {
    if (!predictions || predictions.length === 0) return '';

    // Build headers dynamically based on selected models
    const headers = ['date'];
    const modelNames = {
        gru: 'GRU',
        lstm: 'LSTM',
        hybrid_lstm: 'Hybrid_LSTM',
        hybrid_gru: 'Hybrid_GRU'
    };

    // Add model prediction columns
    state.selectedModels.forEach(model => {
        headers.push(modelNames[model] || model);
    });

    // Add confidence interval columns
    state.selectedModels.forEach(model => {
        const name = modelNames[model] || model;
        headers.push(`${name}_lower`);
        headers.push(`${name}_upper`);
    });

    // Add average prediction column
    headers.push('Average_Prediction');

    const csvRows = [headers.join(',')];

    for (const row of predictions) {
        const values = [row.date];

        // Add predictions for each selected model
        state.selectedModels.forEach(model => {
            values.push(row[model] != null ? row[model] : '');
        });

        // Add confidence intervals for each selected model
        state.selectedModels.forEach(model => {
            const lower = row[`${model}_lower`];
            const upper = row[`${model}_upper`];
            values.push(lower != null ? lower : '');
            values.push(upper != null ? upper : '');
        });

        // Add average prediction
        const ensemble = calculateEnsemblePrice(row);
        values.push(ensemble);

        csvRows.push(values.join(','));
    }

    return csvRows.join('\n');
}

function resetApp() {
    stopTimer();
    
    state.currentStep = 1;
    state.selectedCrypto = null;
    state.selectedCryptoName = null;
    state.predictions = null;
    state.futurePredictions = null;
    state.currentPrice = null;
    state.metrics = null;
    state.abortController = null;
    state.trainingConfig = null;
    state.showConfidence = false;
    state.showPercent = false;
    
    // Destroy charts if exist
    if (state.chart) {
        state.chart.destroy();
        state.chart = null;
    }
    if (state.modelSpecificChart) {
        state.modelSpecificChart.destroy();
        state.modelSpecificChart = null;
    }
    
    // Reset analysis stages UI
    resetStages();
    document.getElementById('elapsed-time').textContent = '00:00';
    
    // Reset chart action buttons
    document.getElementById('toggle-percent')?.classList.remove('active');
    document.getElementById('toggle-confidence')?.classList.remove('active');
    
    // Reset UI
    document.querySelectorAll('.crypto-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('next-step-1').disabled = true;
    
    // Reset checkboxes and re-enable start analysis button
    document.querySelectorAll('input[name="models"]').forEach(cb => cb.checked = true);
    const daysSelect = document.getElementById('days-ahead-select');
    if (daysSelect) {
        daysSelect.value = '7';
    } else {
        const radio = document.querySelector('input[name="days-ahead"][value="7"]');
        if (radio) radio.checked = true;
    }
    state.daysAhead = 7;
    state.selectedModels = ['gru', 'lstm', 'hybrid_lstm', 'hybrid_gru'];
    
    // Reset advanced settings
    document.getElementById('batch-size').value = advancedDefaults.batchSize;
    document.getElementById('epochs').value = advancedDefaults.epochs;
    document.getElementById('learning-rate').value = advancedDefaults.learningRate;
    document.getElementById('hidden-dim').value = advancedDefaults.hiddenDim;
    document.getElementById('start-date').value = advancedDefaults.startDate;
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('end-date').value = today;
    
    // Close advanced settings panel
    document.getElementById('advanced-toggle')?.classList.remove('open');
    document.getElementById('advanced-settings')?.classList.add('hidden');
    
    // Re-enable the start analysis button
    const startBtn = document.getElementById('next-step-2');
    if (startBtn) {
        startBtn.disabled = false;
        startBtn.innerHTML = 'Start Analysis <span class="btn-arrow">→</span>';
    }
    
    // Hide view results button
    document.getElementById('view-results')?.classList.add('hidden');
    
    // Go to step 1
    goToStep(1);
}

function formatPrice(value) {
    if (value === null || value === undefined) return '-';
    
    const absValue = Math.abs(value);
    let decimals;
    
    if (absValue === 0) decimals = 4;
    else if (absValue < 0.00001) decimals = 15;
    else if (absValue < 0.0001) decimals = 12;
    else if (absValue < 0.001) decimals = 10;
    else if (absValue < 0.01) decimals = 9;
    else if (absValue < 0.1) decimals = 8;
    else if (absValue < 1) decimals = 7;
    else if (absValue < 10) decimals = 7;
    else if (absValue < 100) decimals = 6;
    else if (absValue < 1000) decimals = 5;
    else decimals = 4;
    
    return '$' + value.toFixed(decimals);
}


function formatNumber(value) {
    if (value === null || value === undefined) return '-';
    
    const absValue = Math.abs(value);
    
    if (absValue < 0.0001) return value.toExponential(4);
    if (absValue < 1) return value.toFixed(7);
    if (absValue < 1000) return value.toFixed(5);
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatModelName(model) {
    const names = {
        'gru': 'Bi-GRU',
        'lstm': 'Bi-LSTM',
        'hybrid_lstm': 'Hybrid LSTM',
        'hybrid_gru': 'Hybrid GRU'
    };
    return names[model] || model.toUpperCase().replace('_', ' ');
}

/**
 * Calculate ensemble price from available/selected models only
 */
function calculateEnsemblePrice(row) {
    if (!row) return 0;
    
    // Average only the two hybrid models: Hybrid LSTM and Hybrid GRU
    const prices = [];
    if (row.hybrid_lstm != null && !isNaN(row.hybrid_lstm)) {
        prices.push(row.hybrid_lstm);
    }
    if (row.hybrid_gru != null && !isNaN(row.hybrid_gru)) {
        prices.push(row.hybrid_gru);
    }

    if (prices.length === 0) return 0;
    return prices.reduce((sum, p) => sum + p, 0) / prices.length;
}

/**
 * Get first available model price from a row
 */
function getFirstAvailableModelPrice(row) {
    if (!row) return 0;
    
    for (const model of state.selectedModels) {
        if (row[model] != null && !isNaN(row[model])) {
            return row[model];
        }
    }
    
    // Fallback to any available model
    if (row.gru != null && !isNaN(row.gru)) return row.gru;
    if (row.lstm != null && !isNaN(row.lstm)) return row.lstm;
    if (row.hybrid_lstm != null && !isNaN(row.hybrid_lstm)) return row.hybrid_lstm;
    if (row.hybrid_gru != null && !isNaN(row.hybrid_gru)) return row.hybrid_gru;
    
    return 0;
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
        weekday: 'short',
        month: 'short', 
        day: 'numeric' 
    });
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMsg = document.getElementById('toast-message');
    
    toast.className = `toast ${type}`;
    toastMsg.textContent = message;
    toast.classList.add('visible');
    
    setTimeout(() => {
        toast.classList.remove('visible');
    }, 3000);
}
