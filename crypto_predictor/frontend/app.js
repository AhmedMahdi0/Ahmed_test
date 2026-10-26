const API_BASE = '';
const TIMEFRAME_LABELS = { '1h': '1 Hour', '4h': '4 Hours', '1d': '1 Day' };
const TIMEFRAME_ORDER = Object.keys(TIMEFRAME_LABELS);
const DIRECTION = {
    up: { icon: '▲', label: 'UP', cls: 'dir-up' },
    down: { icon: '▼', label: 'DOWN', cls: 'dir-down' },
    neutral: { icon: '●', label: 'NEUTRAL', cls: 'dir-neutral' },
    none: { icon: '–', label: 'NO SIGNAL', cls: 'dir-none' },
};
const COLORS = { up: '#10B981', down: '#EF4444', neutral: '#9CA3AF', primary: '#667eea', muted: '#9CA3AF' };

const state = {
    currentStep: 1,
    selectedCrypto: null,
    selectedCryptoName: null,
    models: [],
    config: null,
    taskId: null,
    pollTimer: null,
    analysisTimer: null,
    analysisStartTime: null,
    analysisRunning: false,
    result: null,
    activeTf: null,
    charts: {},
};

document.addEventListener('DOMContentLoaded', initializeApp);

async function initializeApp() {
    setupEventListeners();
    await Promise.all([loadCryptocurrencies(), loadModels(), loadConfig()]);
}

function setupEventListeners() {
    document.getElementById('next-step-1').addEventListener('click', () => goToStep(2));
    document.getElementById('prev-step-2').addEventListener('click', () => goToStep(1));
    document.getElementById('next-step-2').addEventListener('click', startAnalysis);
    document.getElementById('view-results').addEventListener('click', () => goToStep(4));
    document.getElementById('cancel-training').addEventListener('click', cancelAnalysis);
    document.getElementById('back-to-config').addEventListener('click', () => goToStep(2));
    document.getElementById('refresh-signal').addEventListener('click', refreshSignal);
    document.getElementById('new-prediction').addEventListener('click', resetApp);

    const toggle = document.getElementById('advanced-toggle');
    toggle.addEventListener('click', () => {
        toggle.classList.toggle('open');
        document.getElementById('advanced-settings').classList.toggle('hidden');
    });

    document.querySelectorAll('input[name="timeframes"]').forEach(cb => cb.addEventListener('change', updateStartButton));

    document.querySelectorAll('.progress-steps .step').forEach(step => {
        step.style.cursor = 'pointer';
        step.addEventListener('click', () => {
            const num = parseInt(step.dataset.step, 10);
            if (state.analysisRunning || num === 3) return;
            if (num === 4 && !state.result) return;
            if (num >= 2 && !state.selectedCrypto) return;
            goToStep(num);
        });
    });
}

// ============================================
// Navigation
// ============================================
function goToStep(stepNumber) {
    document.querySelectorAll('.progress-steps .step').forEach(step => {
        const num = parseInt(step.dataset.step, 10);
        step.classList.remove('active', 'completed');
        if (num < stepNumber || (state.result && num === 4 && stepNumber !== 4)) step.classList.add('completed');
        else if (num === stepNumber) step.classList.add('active');
    });
    document.querySelectorAll('.step-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`step-${stepNumber}`).classList.add('active');
    state.currentStep = stepNumber;

    if (stepNumber === 2) {
        document.getElementById('selected-crypto-name').textContent = `${state.selectedCrypto} - ${state.selectedCryptoName}`;
        const sameCrypto = state.result && state.result.symbol === state.selectedCrypto;
        document.getElementById('view-results').classList.toggle('hidden', !sameCrypto);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================
// API
// ============================================
async function apiCall(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* non-JSON body */ }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
}

async function loadCryptocurrencies() {
    try {
        const data = await apiCall('/api/cryptocurrencies');
        const grid = document.getElementById('crypto-grid');
        grid.innerHTML = '';
        data.cryptocurrencies.filter(c => c.enabled).forEach(crypto => {
            const card = document.createElement('div');
            card.className = 'crypto-card';
            card.dataset.symbol = crypto.symbol;
            card.dataset.name = crypto.name;
            card.innerHTML = `<div class="crypto-symbol"></div><div class="crypto-name"></div>`;
            card.querySelector('.crypto-symbol').textContent = crypto.symbol;
            card.querySelector('.crypto-name').textContent = crypto.name;
            card.addEventListener('click', () => {
                document.querySelectorAll('.crypto-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                state.selectedCrypto = crypto.symbol;
                state.selectedCryptoName = crypto.name;
                document.getElementById('next-step-1').disabled = false;
            });
            grid.appendChild(card);
        });
    } catch (error) {
        showToast('Failed to load cryptocurrencies', 'error');
    }
}

async function loadModels() {
    try {
        const data = await apiCall('/api/models');
        state.models = data.models;
        const container = document.getElementById('model-options');
        container.innerHTML = '';
        data.models.forEach(model => {
            const label = document.createElement('label');
            label.className = 'model-option' + (model.available ? '' : ' disabled');
            label.innerHTML = `
                <input type="checkbox" name="models" value="${model.name}"
                    ${model.default && model.available ? 'checked' : ''} ${model.available ? '' : 'disabled'}>
                <span class="model-info"><strong></strong><small></small></span>`;
            label.querySelector('strong').textContent = model.label;
            label.querySelector('small').textContent = model.available ? model.description : 'Requires PyTorch (not installed)';
            label.querySelector('input').addEventListener('change', updateStartButton);
            container.appendChild(label);
        });
        updateStartButton();
    } catch (error) {
        showToast('Failed to load models', 'error');
    }
}

async function loadConfig() {
    try {
        state.config = await apiCall('/api/config');
        const cfg = state.config;
        document.querySelectorAll('.thr-input').forEach(input => {
            input.value = +(cfg.neutral_thresholds[input.dataset.tf] * 100).toFixed(3);
        });
        document.querySelectorAll('#advanced-settings [data-key]').forEach(input => {
            const value = cfg[input.dataset.key];
            if (input.type === 'checkbox') input.checked = !!value;
            else input.value = value;
        });
        const modeValue = cfg.label_mode === 'ternary' ? 'ternary' : (cfg.drop_weak_moves ? 'binary_strong' : 'binary');
        const mode = document.querySelector(`input[name="label-mode"][value="${modeValue}"]`);
        if (mode) mode.checked = true;
        document.getElementById('signal-filter').checked = !!cfg.signal_filter;
        document.getElementById('keep-fraction').value = Math.round(cfg.signal_keep_fraction * 100);
    } catch (error) {
        showToast('Failed to load settings', 'error');
    }
}

function selectedValues(name) {
    return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map(cb => cb.value);
}

function updateStartButton() {
    const btn = document.getElementById('next-step-2');
    const ok = selectedValues('models').length > 0 && selectedValues('timeframes').length > 0;
    btn.disabled = !ok;
    btn.innerHTML = ok ? 'Start Analysis <span class="btn-arrow">→</span>' : 'Select a timeframe and a model';
}

function collectSettings() {
    const mode = selectedValues('label-mode')[0] || 'binary_strong';
    const settings = {
        label_mode: mode === 'ternary' ? 'ternary' : 'binary',
        drop_weak_moves: mode === 'binary_strong',
        signal_filter: document.getElementById('signal-filter').checked,
        neutral_thresholds: {},
    };
    const keep = parseFloat(document.getElementById('keep-fraction').value);
    if (!Number.isNaN(keep)) settings.signal_keep_fraction = Math.min(100, Math.max(5, keep)) / 100;
    document.querySelectorAll('.thr-input').forEach(input => {
        if (input.value !== '') settings.neutral_thresholds[input.dataset.tf] = parseFloat(input.value) / 100;
    });
    document.querySelectorAll('#advanced-settings [data-key]').forEach(input => {
        if (input.type === 'checkbox') settings[input.dataset.key] = input.checked;
        else if (input.tagName === 'SELECT') settings[input.dataset.key] = input.value;
        else if (input.value !== '') settings[input.dataset.key] = parseFloat(input.value);
    });
    return settings;
}

// ============================================
// Analysis
// ============================================
async function startAnalysis() {
    const timeframes = selectedValues('timeframes');
    const models = selectedValues('models');
    if (!timeframes.length || !models.length) return;

    resetProgress(timeframes);
    goToStep(3);
    state.analysisRunning = true;
    startTimer();

    try {
        const data = await apiCall('/api/analyze', {
            method: 'POST',
            body: JSON.stringify({ symbol: state.selectedCrypto, timeframes, models, settings: collectSettings() }),
        });
        state.taskId = data.task_id;
        pollAnalysis(timeframes);
    } catch (error) {
        finishAnalysis(false, error.message);
    }
}

function resetProgress(timeframes) {
    document.getElementById('analyzing-crypto').textContent = state.selectedCrypto;
    document.getElementById('analysis-icon').textContent = '⚙️';
    document.getElementById('analysis-icon').classList.add('spinning');
    document.getElementById('progress-message').textContent = 'Starting...';
    document.getElementById('cancel-training').disabled = false;
    const stages = document.getElementById('analysis-stages');
    stages.innerHTML = timeframes.map(tf => `
        <div class="stage" id="stage-${tf}">
            <div class="stage-status">
                <span class="stage-icon">🕯️</span>
                <span class="stage-name">${TIMEFRAME_LABELS[tf]} candles</span>
            </div>
            <div class="stage-progress"><div class="stage-bar"></div></div>
        </div>`).join('');
}

function pollAnalysis(timeframes) {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
        try {
            const status = await apiCall(`/api/analyze/${state.taskId}`);
            updateProgress(timeframes, status.progress || 0, status.progress_message);
            if (status.status === 'completed') {
                state.result = status.result;
                state.activeTf = orderedTimeframes()[0];
                finishAnalysis(true);
            } else if (status.status === 'failed') {
                finishAnalysis(false, status.error);
            } else if (status.status === 'cancelled') {
                finishAnalysis(false, 'Analysis cancelled');
            } else {
                pollAnalysis(timeframes);
            }
        } catch (error) {
            finishAnalysis(false, error.message);
        }
    }, 1000);
}

function updateProgress(timeframes, progress, message) {
    const share = 100 / timeframes.length;
    timeframes.forEach((tf, i) => {
        const el = document.getElementById(`stage-${tf}`);
        if (!el) return;
        const local = Math.max(0, Math.min(100, (progress - i * share) / share * 100));
        el.classList.toggle('completed', local >= 100);
        el.classList.toggle('active', local > 0 && local < 100);
        el.querySelector('.stage-bar').style.width = `${local}%`;
    });
    if (message) document.getElementById('progress-message').textContent = message;
}

async function cancelAnalysis() {
    if (!state.taskId) return;
    document.getElementById('cancel-training').disabled = true;
    document.getElementById('progress-message').textContent = 'Cancelling...';
    try {
        await apiCall(`/api/analyze/${state.taskId}/cancel`, { method: 'POST' });
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function finishAnalysis(success, error) {
    clearTimeout(state.pollTimer);
    stopTimer();
    state.analysisRunning = false;
    const icon = document.getElementById('analysis-icon');
    icon.classList.remove('spinning');
    icon.textContent = success ? '✅' : '❌';
    if (success) {
        renderResults();
        setTimeout(() => goToStep(4), 400);
    } else {
        document.getElementById('progress-message').textContent = error || 'Analysis failed';
        showToast(error || 'Analysis failed', 'error');
        setTimeout(() => goToStep(2), 1500);
    }
}

function startTimer() {
    state.analysisStartTime = Date.now();
    clearInterval(state.analysisTimer);
    state.analysisTimer = setInterval(() => {
        const s = Math.floor((Date.now() - state.analysisStartTime) / 1000);
        document.getElementById('elapsed-time').textContent =
            `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(state.analysisTimer);
}

async function refreshSignal() {
    if (!state.result) return;
    const tfs = orderedTimeframes();
    try {
        const data = await apiCall(`/api/signal/${state.result.symbol}?timeframes=${tfs.join(',')}`);
        Object.entries(data.signals).forEach(([tf, signal]) => { state.result.timeframes[tf].signal = signal; });
        state.result.generated_at = data.generated_at;
        renderResults();
        const failed = Object.keys(data.errors || {});
        showToast(failed.length ? `Refreshed; failed for ${failed.join(', ')}` : 'Signals refreshed', failed.length ? 'info' : 'success');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// ============================================
// Results
// ============================================
function renderResults() {
    const r = state.result;
    document.getElementById('results-crypto').textContent = r.symbol;
    document.getElementById('results-date').textContent = formatDateTime(r.generated_at);

    const errors = Object.entries(r.errors || {});
    const errBox = document.getElementById('results-errors');
    errBox.classList.toggle('hidden', errors.length === 0);
    errBox.textContent = errors.map(([tf, e]) => `${tf}: ${e}`).join(' · ');

    const tfs = orderedTimeframes();
    document.getElementById('signal-cards').innerHTML = tfs.map(tf => signalCard(tf, r.timeframes[tf])).join('');

    const tabs = document.getElementById('tf-tabs');
    tabs.innerHTML = tfs.map(tf =>
        `<button class="model-tab ${tf === state.activeTf ? 'active' : ''}" data-tf="${tf}">${TIMEFRAME_LABELS[tf]}</button>`).join('');
    tabs.querySelectorAll('.model-tab').forEach(btn => btn.addEventListener('click', () => {
        state.activeTf = btn.dataset.tf;
        tabs.querySelectorAll('.model-tab').forEach(b => b.classList.toggle('active', b === btn));
        renderDetails(btn.dataset.tf);
    }));
    if (!r.timeframes[state.activeTf]) state.activeTf = tfs[0];
    renderDetails(state.activeTf);
}

// The API's JSON object keys come back sorted alphabetically (1d, 1h, 4h).
function orderedTimeframes() {
    return TIMEFRAME_ORDER.filter(tf => state.result && state.result.timeframes[tf]);
}

function filterSummary(res) {
    const sig = res.metrics.ensemble.signals;
    if (!sig || !sig.n) return 'No past signals passed the filter.';
    const skill = sig.directional_accuracy - sig.always_up_same_candles;
    return `Past signals: shown on ${pct(sig.coverage)} of candles, right ${pct(sig.directional_accuracy)} ` +
        `(always-up on the same candles: ${pct(sig.always_up_same_candles)}, ${skill >= 0 ? '+' : ''}${(skill * 100).toFixed(1)} pts)`;
}

function signalCard(tf, res) {
    const s = res.signal;
    const d = DIRECTION[s.direction];
    if (s.direction === 'none') {
        return `
        <div class="signal-card dir-none">
            <div class="signal-top"><span class="signal-tf">${TIMEFRAME_LABELS[tf]}</span>
                <span class="confidence-badge">filtered</span></div>
            <div class="signal-direction"><span class="signal-icon">${d.icon}</span><span class="signal-label">${d.label}</span></div>
            <div class="signal-sub">next candle · closes ${formatDateTime(s.target_time)}</div>
            <p class="signal-none-text">The call for this candle is too weak, so it is not shown.
                Only the strongest ${pct(res.signal_filter.keep_fraction)} of calls become signals.</p>
            <div class="signal-meta">
                <div><span>Price at last close</span><b>${formatPrice(s.price)}</b></div>
                <div><span>Neutral band</span><b>±${(s.threshold * 100).toFixed(2)}%</b></div>
            </div>
            <p class="signal-hist">${filterSummary(res)}</p>
        </div>`;
    }
    const classes = res.label.classes;
    const bars = [...classes].reverse().map(c => `
        <div class="prob-row">
            <span class="prob-label">${DIRECTION[c].label}</span>
            <div class="prob-track"><div class="prob-fill ${DIRECTION[c].cls}" style="width:${(s.probabilities[c] * 100).toFixed(1)}%"></div></div>
            <span class="prob-value">${pct(s.probabilities[c])}</span>
        </div>`).join('');
    const models = Object.entries(s.models).map(([name, m]) =>
        `<span class="vote ${DIRECTION[m.direction].cls}" title="${modelLabel(name)}">${modelLabel(name, true)} ${DIRECTION[m.direction].icon}</span>`).join('');
    const hist = s.historical && s.historical.n > 0
        ? `Past <b>${s.confidence_level}</b>-confidence ${d.label} calls: right ${pct(s.historical.hit_rate)}${
            s.historical.moved_that_way != null ? `, moved that way ${pct(s.historical.moved_that_way)}` : ''} (n=${s.historical.n})`
        : 'No past signals at this confidence level';
    const ens = res.metrics.ensemble;
    return `
        <div class="signal-card ${d.cls}">
            <div class="signal-top">
                <span class="signal-tf">${TIMEFRAME_LABELS[tf]}</span>
                <span class="confidence-badge conf-${s.confidence_level}">${s.confidence_level} confidence</span>
            </div>
            <div class="signal-direction">
                <span class="signal-icon">${d.icon}</span>
                <span class="signal-label">${d.label}</span>
            </div>
            <div class="signal-sub">next candle · closes ${formatDateTime(s.target_time)}</div>
            <div class="prob-bars">${bars}</div>
            <div class="signal-votes">${models}</div>
            <div class="signal-meta">
                <div><span>Price at last close</span><b>${formatPrice(s.price)}</b></div>
                <div><span>Neutral band</span><b>±${(s.threshold * 100).toFixed(2)}%</b></div>
                <div><span>Shown signals right</span><b>${pct(ens.signals.directional_accuracy)}</b></div>
                <div><span>All calls right</span><b>${pct(ens.directional_accuracy)}</b></div>
            </div>
            <p class="signal-hist">${hist}<br>${filterSummary(res)}</p>
        </div>`;
}

function renderDetails(tf) {
    const res = state.result.timeframes[tf];
    const container = document.getElementById('tf-details');
    Object.values(state.charts).forEach(c => c.destroy());
    state.charts = {};

    const dd = res.data;
    const balance = Object.entries(dd.class_balance).map(([c, v]) => `${DIRECTION[c].label} ${pct(v)}`).join(' · ');
    const bt = res.backtest;

    container.innerHTML = `
        <div class="data-summary">
            <span><b>Source</b> ${dd.source}</span>
            <span><b>Candles</b> ${dd.candles.toLocaleString()}</span>
            <span><b>Features</b> ${dd.features}</span>
            <span><b>Out-of-sample</b> ${formatDate(dd.test_start)} → ${formatDate(dd.end)} (${dd.test_candles.toLocaleString()} candles, ${res.walk_forward.splits} folds)</span>
            <span><b>Actual mix</b> ${balance}</span>
            <span><b>Feature groups</b> ${groupSummary(dd.feature_groups)}</span>
            <span><b>Training</b> ${res.training.drop_weak_moves
                ? `strong moves only (${pct(res.training.strong_move_share)} of candles)` : 'all candles'}</span>
            <button class="btn btn-secondary btn-sm" id="export-tf">📥 CSV</button>
        </div>
        ${(dd.notes || []).length ? `<div class="data-notes">${dd.notes.map(n => `<div>ℹ️ ${escapeHtml(n)}</div>`).join('')}</div>` : ''}

        <h4 class="detail-title">Model scoreboard (out-of-sample, walk-forward)</h4>
        <div class="table-container">${metricsTable(res)}</div>
        <p class="metrics-note">💡 A model is only useful if it clearly beats the <i>baseline</i> rows.
            <b>Shown signals</b>: the calls that passed the weak-signal filter. Compare them with
            <b>always-up on the same candles</b>: a filter can look accurate just by picking rising stretches.</p>

        <div class="detail-grid">
            <div class="detail-card">
                <h4 class="detail-title">Backtest: trading shown signals only (fees ${bt.fee_bps} bps/side)</h4>
                <div class="bt-stats">
                    ${btStat('Strategy', signedPct(bt.total_return), bt.total_return)}
                    ${btStat('Buy & hold', signedPct(bt.buy_hold_return), bt.buy_hold_return)}
                    ${btStat('Sharpe', bt.sharpe == null ? '--' : bt.sharpe.toFixed(2), bt.sharpe)}
                    ${btStat('Max drawdown', signedPct(bt.max_drawdown), -1)}
                    ${btStat('Trades', bt.trades.toLocaleString())}
                    ${btStat('Win rate', pct(bt.win_rate))}
                </div>
                <div class="chart-wrapper small"><canvas id="equity-chart"></canvas></div>
            </div>
            <div class="detail-card">
                <h4 class="detail-title">Confusion matrix (ensemble, all calls before the filter)</h4>
                ${confusionMatrix(res)}
                <h4 class="detail-title">Top features</h4>
                ${featureBars(res.feature_importance)}
            </div>
        </div>

        <h4 class="detail-title">Recent out-of-sample signals (shown signals only)</h4>
        <div class="chart-wrapper"><canvas id="history-chart"></canvas></div>
        <div class="table-container">${historyTable(res)}</div>
    `;
    document.getElementById('export-tf').addEventListener('click', () => {
        window.location.href = `${API_BASE}/api/export/${state.result.symbol}/${tf}`;
    });
    renderEquityChart(res);
    renderHistoryChart(res);
}

function metricsTable(res) {
    const rows = Object.entries(res.metrics);
    const best = key => Math.max(...rows.filter(([n]) => !n.startsWith('baseline')).map(([, m]) => m[key] ?? -1));
    const bests = { accuracy: best('accuracy'), balanced_accuracy: best('balanced_accuracy'),
        f1_macro: best('f1_macro'), directional_accuracy: best('directional_accuracy') };
    const cell = (m, key) => `<td class="${m[key] != null && m[key] === bests[key] ? 'best' : ''}">${pct(m[key])}</td>`;
    const shown = m => m.signals
        ? `${pct(m.signals.directional_accuracy)} <small>(${pct(m.signals.coverage)} of candles)</small>` : '--';
    const upSame = m => m.signals ? pct(m.signals.always_up_same_candles) : '';
    const order = ([a], [b]) => rank(a) - rank(b);
    return `<table class="predictions-table metrics-table">
        <thead><tr><th>Model</th><th>Accuracy</th><th>Balanced acc.</th><th>Macro F1</th>
            <th>Directional acc.</th><th>Shown signals right</th><th>Always-up, same candles</th><th>Backtest</th></tr></thead>
        <tbody>${rows.sort(order).map(([name, m]) => `
            <tr class="${name.startsWith('baseline') ? 'baseline-row' : ''} ${name === 'ensemble' ? 'ensemble-row' : ''}">
                <td>${modelLabel(name)}</td>
                ${cell(m, 'accuracy')}${cell(m, 'balanced_accuracy')}${cell(m, 'f1_macro')}${cell(m, 'directional_accuracy')}
                <td>${shown(m)}</td><td>${upSame(m)}</td>
                <td>${name === 'ensemble' ? signedPct(res.backtest.total_return) : (m.backtest_return != null ? signedPct(m.backtest_return) : '')}</td>
            </tr>`).join('')}
        </tbody></table>`;
}

function rank(name) {
    if (name === 'ensemble') return 0;
    if (name.startsWith('baseline')) return 100 + name.length;
    return 10 + Math.max(0, state.models.findIndex(m => m.name === name));
}

function confusionMatrix(res) {
    const classes = res.label.classes;
    const cm = res.metrics.ensemble.confusion_matrix;
    const max = Math.max(...cm.flat());
    return `<table class="confusion">
        <thead><tr><th>actual ↓ / predicted →</th>${classes.map(c => `<th>${DIRECTION[c].label}</th>`).join('')}</tr></thead>
        <tbody>${cm.map((row, i) => `<tr><th>${DIRECTION[classes[i]].label}</th>${row.map((v, j) =>
            `<td class="${i === j ? 'diag' : ''}" style="--alpha:${(v / max).toFixed(2)}">${v.toLocaleString()}</td>`).join('')}</tr>`).join('')}
        </tbody></table>`;
}

function featureBars(items) {
    if (!items || !items.length) return '<p class="metrics-note">Feature importance is available when Gradient Boosting is selected.</p>';
    const max = items[0].importance;
    return `<div class="feature-bars">${items.slice(0, 10).map(f => `
        <div class="feature-row"><span class="feature-name">${f.feature}</span>
            <div class="prob-track"><div class="prob-fill feature-fill" style="width:${(f.importance / max * 100).toFixed(1)}%"></div></div>
            <span class="prob-value">${pct(f.importance)}</span></div>`).join('')}</div>`;
}

function historyTable(res) {
    const h = res.history;
    const rows = h.time.map((t, i) => ({ t, close: h.close[i], pred: h.signal[i], actual: h.actual[i], conf: h.confidence[i] }))
        .filter(r => r.pred !== 'none').slice(-15).reverse();
    if (!rows.length) return '<p class="metrics-note">No signals passed the filter in this period.</p>';
    return `<table class="predictions-table">
        <thead><tr><th>Candle</th><th>Close</th><th>Signal</th><th>Confidence</th><th>Actual</th><th></th></tr></thead>
        <tbody>${rows.map(r => `<tr>
            <td>${formatDateTime(r.t)}</td><td>${formatPrice(r.close)}</td>
            <td class="${DIRECTION[r.pred].cls}-text">${DIRECTION[r.pred].icon} ${DIRECTION[r.pred].label}</td>
            <td>${pct(r.conf)}</td>
            <td class="${DIRECTION[r.actual].cls}-text">${DIRECTION[r.actual].icon} ${DIRECTION[r.actual].label}</td>
            <td>${r.pred === r.actual ? '✅' : '❌'}</td></tr>`).join('')}
        </tbody></table>`;
}

function groupSummary(groups) {
    const names = { price: 'price', btc: 'BTC', macro: 'macro', futures: 'futures',
        higher_timeframe: 'higher TF', support_resistance: 'S/R' };
    return Object.entries(groups || {}).filter(([, n]) => n > 0).map(([g, n]) => `${names[g] || g} ${n}`).join(' · ');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function btStat(label, value, sign) {
    const cls = sign == null ? '' : (sign > 0 ? 'pos' : (sign < 0 ? 'neg' : ''));
    return `<div class="bt-stat"><span>${label}</span><b class="${cls}">${value}</b></div>`;
}

function timeScale(tf) {
    return { type: 'time', time: { unit: tf === '1d' ? 'month' : 'day' }, grid: { display: false } };
}

function renderEquityChart(res) {
    const eq = res.backtest.equity_curve;
    state.charts.equity = new Chart(document.getElementById('equity-chart'), {
        type: 'line',
        data: {
            labels: eq.time,
            datasets: [
                { label: 'Strategy', data: eq.strategy, borderColor: COLORS.primary, borderWidth: 2, pointRadius: 0, tension: 0.1 },
                { label: 'Buy & hold', data: eq.buy_hold, borderColor: COLORS.muted, borderWidth: 1.5, pointRadius: 0, borderDash: [4, 4] },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
            scales: { x: timeScale(state.activeTf), y: { ticks: { callback: v => `${((v - 1) * 100).toFixed(0)}%` } } },
            plugins: { tooltip: { callbacks: { label: c => `${c.dataset.label}: ${signedPct(c.parsed.y - 1)}` } } },
        },
    });
}

function renderHistoryChart(res) {
    const h = res.history;
    const points = dir => h.time.map((t, i) => (h.signal[i] === dir ? { x: t, y: h.close[i] } : null)).filter(Boolean);
    state.charts.history = new Chart(document.getElementById('history-chart'), {
        data: {
            datasets: [
                { type: 'line', label: 'Close', data: h.time.map((t, i) => ({ x: t, y: h.close[i] })),
                  borderColor: '#4B5563', borderWidth: 1.5, pointRadius: 0, order: 3 },
                { type: 'scatter', label: 'Signal UP', data: points('up'), backgroundColor: COLORS.up,
                  pointStyle: 'triangle', pointRadius: 5, order: 1 },
                { type: 'scatter', label: 'Signal DOWN', data: points('down'), backgroundColor: COLORS.down,
                  pointStyle: 'triangle', rotation: 180, pointRadius: 5, order: 2 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { x: timeScale(state.activeTf), y: { ticks: { callback: v => formatPrice(v) } } },
        },
    });
}

function resetApp() {
    clearTimeout(state.pollTimer);
    stopTimer();
    Object.values(state.charts).forEach(c => c.destroy());
    Object.assign(state, { selectedCrypto: null, selectedCryptoName: null, taskId: null, result: null,
        activeTf: null, charts: {}, analysisRunning: false });
    document.querySelectorAll('.crypto-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('next-step-1').disabled = true;
    goToStep(1);
}

// ============================================
// Formatting
// ============================================
function pct(v) {
    return v == null ? '--' : `${(v * 100).toFixed(1)}%`;
}

function signedPct(v) {
    return v == null ? '--' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
}

function formatPrice(v) {
    if (v == null) return '--';
    const digits = v >= 1000 ? 0 : v >= 1 ? 2 : v >= 0.01 ? 4 : 6;
    return '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatDateTime(iso) {
    if (!iso) return '--';
    return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatDate(iso) {
    return iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '--';
}

function modelLabel(name, short = false) {
    const names = {
        ensemble: ['Ensemble', 'ENS'], gbm: ['Gradient Boosting', 'GBM'], logreg: ['Logistic Regression', 'LR'],
        gru: ['GRU', 'GRU'], lstm: ['LSTM', 'LSTM'], transformer: ['Transformer', 'TF'],
        baseline_majority: ['Baseline: majority class', 'MAJ'], baseline_persistence: ['Baseline: same as last candle', 'PERS'],
        baseline_always_up: ['Baseline: always up', 'UP'],
    };
    return (names[name] || [name, name])[short ? 1 : 0];
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.className = `toast ${type}`;
    document.getElementById('toast-message').textContent = message;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 3500);
}
