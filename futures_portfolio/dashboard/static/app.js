/**
 * Prosperous Bot Dashboard — Main JS
 * Dashboard page logic: auto-refresh, PM2 control, alerts
 */

// ─── State ────────────────────────────────────────────────────

let refreshTimer = null;

// ─── Init ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadOverview();
    startAutoRefresh();
});

function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(loadOverview, REFRESH_SEC * 1000);
}

// ─── API calls ────────────────────────────────────────────────

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

// ─── Load overview ────────────────────────────────────────────

async function loadOverview() {
    const data = await apiGet('/api/overview');
    if (!data) return;

    updateSummary(data.summary, data.supervisor);
    updateAlerts(data.alerts);
    updateBotsTable('real-bots-table', data.real_bots);
    updateBotsTable('paper-bots-table', data.paper_bots);
    updatePm2Table(data.pm2_processes);
}

// ─── Update summary ───────────────────────────────────────────

function updateSummary(summary, supervisor) {
    // Supervisor
    const supEl = document.getElementById('supervisor-status');
    if (supEl) {
        supEl.textContent = supervisor;
        supEl.className = 'card-value ' + (supervisor === 'online' ? 'pnl-positive' : 'pnl-negative');
    }

    // Total TPV
    const tpvEl = document.getElementById('total-tpv');
    if (tpvEl) tpvEl.textContent = formatUsd(summary.total_tpv);

    // Total PnL
    const pnlEl = document.getElementById('total-pnl');
    if (pnlEl) {
        pnlEl.textContent = `${formatUsd(summary.total_pnl)} (${summary.total_pnl_pct > 0 ? '+' : ''}${summary.total_pnl_pct}%)`;
        pnlEl.className = 'card-value ' + (summary.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative');
    }

    // Bots count
    const botsEl = document.getElementById('bots-count');
    if (botsEl) botsEl.textContent = `${summary.real_count}R / ${summary.paper_count}P`;
}

// ─── Update alerts ────────────────────────────────────────────

function updateAlerts(alerts) {
    const container = document.getElementById('alerts-container');
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = alerts.map(a => `
        <div class="alert alert-${a.level === 'critical' ? 'danger' : 'warning'}">
            ${a.level === 'critical' ? '🚨' : '⚠️'} ${escapeHtml(a.message)}
        </div>
    `).join('');
}

// ─── Update bots table ────────────────────────────────────────

function updateBotsTable(tableId, bots) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    if (!tbody) return;

    if (!bots || bots.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">Нет данных</td></tr>';
        return;
    }

    tbody.innerHTML = bots.map(bot => {
        const statusClass = bot.status === 'ok' ? 'badge-ok' :
                           bot.status === 'stale' ? 'badge-stale' :
                           bot.status === 'pnl_guard' ? 'badge-pnl-guard' : 'badge-offline';
        const pnlClass = bot.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        const hbText = bot.heartbeat_ok ? '✅' : `⚠️ ${bot.heartbeat_age_sec || '?'}s`;

        return `
            <tr>
                <td><strong><a href="/ticker/${bot.mode}/${bot.ticker}" style="color:var(--text-primary);text-decoration:none">${bot.ticker}</a></strong></td>
                <td>${formatUsd(bot.tpv)}</td>
                <td class="${pnlClass}">${formatUsd(bot.pnl)} (${bot.pnl_pct > 0 ? '+' : ''}${bot.pnl_pct}%)</td>
                <td>${bot.cycles || 0}</td>
                <td>${hbText}</td>
                <td><span class="badge ${statusClass}">${bot.status}</span></td>
                <td>
                    <a href="/ticker/${bot.mode}/${bot.ticker}" class="btn btn-sm btn-outline">📊</a>
                    <a href="/logs/${bot.mode}/${bot.ticker}" class="btn btn-sm btn-outline">📜</a>
                </td>
            </tr>
        `;
    }).join('');
}

// ─── Update PM2 table ─────────────────────────────────────────

function updatePm2Table(processes) {
    const tbody = document.querySelector('#pm2-table tbody');
    if (!tbody) return;

    if (!processes || processes.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">PM2 не отвечает</td></tr>';
        return;
    }

    tbody.innerHTML = processes.map(p => {
        const statusClass = p.status === 'online' ? 'badge-ok' : 'badge-offline';
        const memMb = p.memory ? (p.memory / 1024 / 1024).toFixed(1) : '—';

        return `
            <tr>
                <td>${escapeHtml(p.name)}</td>
                <td><span class="badge ${statusClass}">${p.status}</span></td>
                <td>${p.restarts || 0}</td>
                <td>${p.cpu || 0}%</td>
                <td>${memMb} MB</td>
            </tr>
        `;
    }).join('');
}

// ─── PM2 Control ──────────────────────────────────────────────

function pm2Action(action, target) {
    const actionNames = {
        pm2_start: 'Запустить',
        pm2_stop: 'Остановить',
        pm2_restart: 'Перезапустить',
        pm2_delete: 'Удалить',
    };

    showConfirm(
        `${actionNames[action] || action}: ${target}`,
        `Вы уверены? Это действие ${action === 'pm2_stop' || action === 'pm2_delete' ? 'остановит' : 'перезапустит'} процесс.`,
        async () => {
            const result = await apiPost('/api/control/pm2', { action, target });
            if (result.success) {
                setTimeout(loadOverview, 2000);
            } else {
                alert('Ошибка: ' + (result.error || result.stderr || 'Unknown'));
            }
        }
    );
}

// ─── Emergency Stop ───────────────────────────────────────────

function emergencyStop(ticker, mode) {
    showConfirm(
        `🚨 ЭКСТРЕННАЯ ОСТАНОВКА: ${mode} ${ticker}`,
        `Бот будет остановлен. PM2 процесс удалён. Позиции на бирже НЕ закрываются автоматически — убедитесь что они закрыты вручную или через Binance.`,
        async () => {
            const result = await apiPost('/api/control/emergency-stop', { ticker, mode });
            if (result.success) {
                setTimeout(loadOverview, 2000);
            } else {
                alert('Ошибка: ' + (result.error || 'Unknown'));
            }
        }
    );
}

// ─── Modal ────────────────────────────────────────────────────

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

// ─── Nav toggle (mobile) ──────────────────────────────────────

function toggleNav() {
    document.querySelector('.nav-links').classList.toggle('show');
}

// ─── Helpers ──────────────────────────────────────────────────

function formatUsd(value) {
    if (value === null || value === undefined) return '—';
    return '$' + Number(value).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
