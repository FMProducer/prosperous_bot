"""Auto-add project root to sys.path for package-style imports."""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Re-export swarm_manager functions for backward compatibility
# (allows supervisor.get_running_bots_info() etc.)
from futures_portfolio.supervisor.swarm_manager import (
    get_running_bots_info,
    start_bot,
    stop_bot,
    manage_swarm,
    calculate_bot_score,
    _calc_rotation_score,
    selective_merge_incubator,
    get_bot_efficiency,
    reset_bot_state_files,
    enforce_swarm_consistency,
    _ensure_real_bots_alive,
    get_pm2_processes,
    reconcile_swarm_state,
    enforce_invariant_gate,
)
