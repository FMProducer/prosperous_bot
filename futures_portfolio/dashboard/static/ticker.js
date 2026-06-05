/**
 * Ticker detail page JS
 */

let refreshTimer = null;
let logTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    loadTickerDetails();
    loadLogs();
    refreshTimer = setInterval(loadTickerDetails, REFRESH_SEC * 1000);
    logTimer = setInterval(loadLogs, REFRESH_SEC * 1000);
});

async function apiGet(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (e) {
        console.error('API error:', e);
        return null;
    }
}

async function apiPost(url, data) {
    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': CSRF_TOKEN,
            },
            body: JSON.stringify(data),
        });
        return await resp.json();
    } catch (e) {
        console.error('API error:', e);
        return { error: e.message };
    }
}

async function loadTickerDetails() {
    const data = await apiGet(`/api/ticker/${MODE}/${TICKER}`);
    if (!data || data.error) {
        document.getElementById('ticker-stats').innerHTML =
            `<div class="alert alert-danger">${data?.error || 'Ошибка загрузки'}</div>`;
        return;
    }

    updateStats(data);
    updatePositions(data.shadow);
}

function updateStats(data) {
    const state = data.state || {};
    const container = document.getElementById('ticker-stats');

    const tpv = state.tpv || 0;
    const initial = state.initial_capital || 0;
    const pnl = tpv - initial;
    const pnlPct = initial > 0 ? (pnl / initial * 100) : 0;
    const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
    const hb = data.heartbeat || {};

    container.innerHTML = `
        <div class="card card-summary">
            <div class="card-icon">💰</div>
            <div class="card-body"><h3>TPV</h3><p class="card-value">$${fmt(tpv)}</p></div>
        </div>
        <div class="card card-summary">
            <div class="card-icon">📈</div>
            <div class="card-body"><h3>PnL</h3><p class="card-value ${pnlClass}">$${fmt(pnl)} (${pnlPct > 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</p></div>
        </div>
        <div class="card card-summary">
            <div class="card-icon">🔄</div>
            <div class="card-body"><h3>Циклы</h3><p class="card-value">${state.rebalance_cycles || state.cycles || 0}</p></div>
        </div>
        <div class="card card-summary">
            <div class="card-icon">${hb.ok ? '✅' : '⚠️'}</div>
            <div class="card-body"><h3>Heartbeat</h3><p class="card-value ${hb.ok ? 'pnl-positive' : 'pnl-negative'}">${hb.ok ? 'OK' : (hb.age_sec ? hb.age_sec + 's' : 'N/A')}</p></div>
        </div>
    `;
}

function updatePositions(shadow) {
    const container = document.getElementById('positions-data');
    if (!shadow) {
        container.innerHTML = '<p style="color:var(--text-muted)">Нет данных shadow</p>';
        return;
    }

    const positions = shadow.positions || shadow.pos || {};
    const entries = Object.entries(positions);

    if (entries.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">Нет открытых позиций</p>';
        return;
    }

    container.innerHTML = `
        <div class="table-responsive">
        <table class="table">
            <thead><tr><th>Сторона</th><th>Количество</th><th>Цена входа</th><th>Лицо (USDT)</th></tr></thead>
            <tbody>
                ${entries.map(([side, pos]) => `
                    <tr>
                        <td><span class="badge ${side.toUpperCase() === 'LONG' || side.toUpperCase() === 'L' ? 'badge-ok' : 'badge-stale'}">${side}</span></td>
                        <td>${pos.qty || pos.amount || '—'}</td>
                        <td>${pos.entry_price || pos.price || '—'}</td>
                        <td>$${fmt((pos.qty || 0) * (pos.mark_price || pos.price || 0))}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
        </div>
    `;
}

async function loadLogs() {
    const data = await apiGet(`/api/logs/${MODE}/${TICKER}?lines=150`);
    const el = document.getElementById('log-content');
    if (!el) return;

    if (!data || data.error) {
        el.textContent = 'Ошибка загрузки логов: ' + (data?.error || 'Unknown');
        return;
    }

    el.textContent = data.lines?.join('\n') || 'Лог пуст';

    // Auto-scroll to bottom
    const viewer = document.getElementById('log-viewer');
    if (viewer) viewer.scrollTop = viewer.scrollHeight;
}

// Control actions (shared with app.js but duplicated for standalone)
function pm2Action(action, target) {
    const names = { pm2_restart: 'Перезапустить', pm2_stop: 'Остановить', pm2_start: 'Запустить' };
    showConfirm(`${names[action] || action}: ${target}`, 'Вы уверены?', async () => {
        const result = await apiPost('/api/control/pm2', { action, target });
        if (result.success) setTimeout(loadTickerDetails, 2000);
        else alert('Ошибка: ' + (result.error || result.stderr || 'Unknown'));
    });
}

function emergencyStop(ticker, mode) {
    showConfirm(`🚨 ЭКСТРЕННАЯ ОСТАНОВКА: ${mode} ${ticker}`,
        'Бот остановлен, PM2 процесс удалён. Позиции НЕ закрываются автоматически.',
        async () => {
            const result = await apiPost('/api/control/emergency-stop', { ticker, mode });
            if (result.success) setTimeout(loadTickerDetails, 2000);
            else alert('Ошибка: ' + (result.error || 'Unknown'));
        });
}

let confirmCallback = null;
function showConfirm(title, message, callback) {
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-message').textContent = message;
    confirmCallback = callback;
    document.getElementById('confirm-modal').classList.add('show');
}
function closeModal() {
    document.getElementById('confirm-modal').classList.remove('show');
    confirmCallback = null;
}
document.getElementById('confirm-btn')?.addEventListener('click', () => {
    if (confirmCallback) confirmCallback();
    closeModal();
});
function toggleNav() {
    document.querySelector('.nav-links').classList.toggle('show');
}

function fmt(v) {
    return Number(v || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
