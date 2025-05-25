import logging
import json
from typing import Dict, Any
from datetime import datetime, timezone

class Config:
    """Utility class to load & persist bot configuration (JSON)."""

    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config: Dict[str, Any] = self._load()

    # ---------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_file} not found.")
            return {}
        except json.JSONDecodeError as err:
            logging.error(f"Invalid JSON format in {self.config_file}: {err}")
            return {}

    def _save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    # ---------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self._save()

    # --- Helpers for often‑used sections ---------------------
    def get_api_keys(self) -> Dict[str, str]:
        return {
            'gate_api_key'   : self.get('gate_api_key'),
            'gate_api_secret': self.get('gate_api_secret'),
        }

    def get_trading_params(self) -> Dict[str, Any]:
        return {
            'fixed_order_size'        : self.get('fixed_order_size'),
            'balance_check_pairs'     : self.get('balance_check_pairs'),
            'binance_pairs'           : self.get('binance_pairs'),
            'interval'                : self.get('interval'),
            'min_price_change_percent': self.get('min_price_change_percent'),
        }

    def get_rebalance_params(self) -> Dict[str, Any]:
        return {
            'min_usdt_balance'  : self.get('min_usdt_balance'),
            'rebalance_threshold': self.get('rebalance_threshold'),
            'ratio_tolerance'   : self.get('ratio_tolerance'),
            'max_deviation'     : self.get('max_deviation'),
            'rebalance_interval': self.get('rebalance_interval'),
            'rebalance_delay'   : self.get('rebalance_delay'),
            'default_commission': self.get('default_commission'),
            'min_order_size'    : self.get('min_order_size'),
            'min_portfolio_value': self.get('min_portfolio_value'),
            'asset_distribution': self.get('asset_distribution'),
            'rebalance_tokens'  : self.get('rebalance_tokens'),
            'balance_check_pairs': self.get('balance_check_pairs'),
        }

    # --- Tracking last common signal ------------------------
    def get_last_common_signal(self):
        sig = self.get('last_common_signal')
        t   = self.get('last_common_signal_time')
        ts  = datetime.fromisoformat(t) if t else None
        return sig, ts

    def set_last_common_signal(self, signal: str):
        self.config['last_common_signal'] = signal
        self.config['last_common_signal_time'] = datetime.now(timezone.utc).isoformat()
        self._save()
