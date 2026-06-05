/**
 * History page JS
 */

document.addEventListener('DOMContentLoaded', loadHistory);

async function loadHistory() {
    try {
        const resp = await fetch('/api/history');
        const data = await resp.json();
        const tbody = document.querySelector('#history-table tbody');
        if (!tbody) return;

        if (!data.history || data.history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">Нет данных</td></tr>';
            return;
        }

        tbody.innerHTML = data.history.map(row => {
            const pnlClass = row.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            return `<tr>
                <td><strong>${row.ticker}</strong></td>
                <td>${row.date}</td>
                <td>$${fmt(row.tpv)}</td>
                <td>$${fmt(row.initial)}</td>
                <td class="${pnlClass}">$${fmt(row.pnl)}</td>
                <td class="${pnlClass}">${row.pnl_pct > 0 ? '+' : ''}${row.pnl_pct}%</td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.error('History load error:', e);
    }
}

function fmt(v) {
    return Number(v || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
