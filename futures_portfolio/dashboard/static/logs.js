/**
 * Logs page JS — real-time log viewer with auto-scroll
 */

let autoScroll = true;
let logTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    loadLogs();
    logTimer = setInterval(loadLogs, REFRESH_SEC * 1000);
});

async function loadLogs() {
    try {
        const resp = await fetch(`/api/logs/${MODE}/${TICKER}?lines=200`);
        const data = await resp.json();
        const el = document.getElementById('log-content');
        const info = document.getElementById('log-info');
        if (!el) return;

        if (data.error) {
            el.textContent = 'Ошибка: ' + data.error;
            return;
        }

        el.textContent = data.lines?.join('\n') || 'Лог пуст';
        if (info) info.textContent = `${data.total || '?'} строк | ${data.file || ''}`;

        if (autoScroll) {
            const viewer = document.getElementById('log-viewer');
            if (viewer) viewer.scrollTop = viewer.scrollHeight;
        }
    } catch (e) {
        console.error('Logs load error:', e);
    }
}

function refreshLogs() {
    loadLogs();
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const btn = document.getElementById('autoscroll-btn');
    if (btn) btn.textContent = `📜 Авто-скролл: ${autoScroll ? 'ВКЛ' : 'ВЫКЛ'}`;
}

function toggleNav() {
    document.querySelector('.nav-links').classList.toggle('show');
}
