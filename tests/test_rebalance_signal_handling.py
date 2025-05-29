import unittest
import pandas as pd
from pathlib import Path
import shutil
import logging
import os
from datetime import timezone, timedelta

# Ensure the src directory is in the Python path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from prosperous_bot.rebalance_backtester import load_signal_data, run_backtest

# Helper function to create dummy CSV files for testing
def create_dummy_csv(filepath: Path, data_dict: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data_dict)
    df.to_csv(filepath, index=False)
    return str(filepath)

class TestSignalHandling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path("tests/test_data_signal_handling")
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        # Suppress most logging output during tests unless specifically testing for logs
        # However, rebalance_backtester itself might set logging level based on config.
        # So, we will manage verbosity per test or rely on config.
        # For assertLogs, the default logger needs to be at a level that allows the message.
        logging.getLogger().setLevel(logging.INFO) # Ensure INFO messages can be caught by assertLogs

    @classmethod
    def tearDownClass(cls):
        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)
        logging.getLogger().setLevel(logging.WARNING) # Reset logger level


    def tearDown(self):
        # Clean up any specific files created within a test if necessary,
        # but setUpClass/tearDownClass handle the main directory.
        # Specifically, remove temp_reports_dir if created by a test
        if hasattr(self, 'temp_reports_dir') and self.temp_reports_dir.exists():
            shutil.rmtree(self.temp_reports_dir)


    # --- Tests for load_signal_data ---

    def test_load_signal_data_valid_file(self):
        signal_file_path = self.test_data_dir / "valid_signals.csv"
        data = {
            'timestamp': ['2023-01-01T10:00:00Z', '2023-01-01T09:00:00', '2023-01-01T11:00:00+01:00'],
            'signal': ['BUY', 'SELL', 'NEUTRAL'],
            'extra_col': [1, 2, 3]
        }
        create_dummy_csv(signal_file_path, data)
        
        df_signals = load_signal_data(str(signal_file_path))
        
        self.assertIsNotNone(df_signals)
        self.assertEqual(list(df_signals.columns), ['timestamp', 'signal'])
        self.assertEqual(len(df_signals), 3)
        
        # Check timestamp conversion to UTC and sorting
        # Timestamps: 2023-01-01T09:00:00 (naive, becomes UTC), 
        #             2023-01-01T10:00:00Z (UTC), 
        #             2023-01-01T11:00:00+01:00 (is 2023-01-01T10:00:00 UTC)
        # Expected sorted distinct UTC timestamps:
        ts1 = pd.Timestamp('2023-01-01T09:00:00Z')
        ts2 = pd.Timestamp('2023-01-01T10:00:00Z')
        
        self.assertEqual(df_signals['timestamp'].iloc[0], ts1)
        self.assertEqual(df_signals['timestamp'].iloc[1], ts2) # Second 10:00Z entry
        self.assertEqual(df_signals['timestamp'].iloc[2], ts2) # Could be the other 10:00Z entry
        self.assertTrue(all(df_signals['timestamp'].dt.tz == timezone.utc))

        # Check signal values after sorting by timestamp
        # Corresponding signals to sorted unique timestamps:
        # 09:00:00Z -> SELL
        # 10:00:00Z -> BUY or NEUTRAL (order not guaranteed for identical timestamps)
        self.assertEqual(df_signals[df_signals['timestamp'] == ts1]['signal'].iloc[0], 'SELL')
        self.assertTrue(df_signals[df_signals['timestamp'] == ts2]['signal'].isin(['BUY', 'NEUTRAL']).all())
        self.assertEqual(len(df_signals[df_signals['timestamp'] == ts2]), 2)


    def test_load_signal_data_file_not_found(self):
        with self.assertLogs(level='WARNING') as log:
            df_signals = load_signal_data(str(self.test_data_dir / "non_existent_file.csv"))
            self.assertIsNone(df_signals)
            self.assertTrue(any("Signal data file not found" in message for message in log.output))


    def test_load_signal_data_empty_file(self):
        signal_file_path = self.test_data_dir / "empty_signals.csv"
        with open(signal_file_path, 'w') as f:
            f.write("") # Truly empty
        
        with self.assertLogs(level='WARNING') as log:
            df_signals = load_signal_data(str(signal_file_path))
            self.assertIsNone(df_signals) 
            self.assertTrue(any("empty" in message.lower() for message in log.output))
        # Test for file that is just headers (e.g. from empty DataFrame save)
        signal_file_path_headers = self.test_data_dir / "headers_only_signals.csv"
        create_dummy_csv(signal_file_path_headers, pd.DataFrame(columns=['timestamp', 'signal']))
        with self.assertLogs(level='WARNING') as log_headers:
            df_signals_headers = load_signal_data(str(signal_file_path_headers))
            self.assertIsNone(df_signals_headers)
            self.assertTrue(any("empty" in message.lower() for message in log_headers.output))


    def test_load_signal_data_missing_timestamp_column(self):
        signal_file_path = self.test_data_dir / "missing_ts_signals.csv"
        create_dummy_csv(signal_file_path, {'signal': ['BUY', 'SELL'], 'data':[1,2]})
        with self.assertLogs(level='ERROR') as log:
            df_signals = load_signal_data(str(signal_file_path))
            self.assertIsNone(df_signals)
            self.assertTrue(any("must contain 'timestamp' and 'signal' columns" in message for message in log.output))

    def test_load_signal_data_missing_signal_column(self):
        signal_file_path = self.test_data_dir / "missing_sig_signals.csv"
        create_dummy_csv(signal_file_path, {'timestamp': ['2023-01-01T10:00:00Z'], 'data':[1]})
        with self.assertLogs(level='ERROR') as log:
            df_signals = load_signal_data(str(signal_file_path))
            self.assertIsNone(df_signals)
            self.assertTrue(any("must contain 'timestamp' and 'signal' columns" in message for message in log.output))

    def test_load_signal_data_invalid_timestamp_format(self):
        signal_file_path = self.test_data_dir / "invalid_ts_format.csv"
        create_dummy_csv(signal_file_path, {'timestamp': ['not-a-date'], 'signal': ['BUY']})
        with self.assertLogs(level='ERROR') as log: 
            df_signals = load_signal_data(str(signal_file_path))
            self.assertIsNone(df_signals) 
            self.assertTrue(any("Error loading or processing signal data" in message for message in log.output))

    # --- Tests for run_backtest signal-based holding logic ---
    def _get_base_params(self, main_asset="TESTASSET", apply_signal_logic_value="OMIT_KEY"):
        # apply_signal_logic_value: True, False, or "OMIT_KEY"
        self.temp_reports_dir = self.test_data_dir / f"temp_reports_{main_asset}_{str(apply_signal_logic_value)}"
        if self.temp_reports_dir.exists(): # Clean up from previous identical test case if any
             shutil.rmtree(self.temp_reports_dir)
        self.temp_reports_dir.mkdir(parents=True, exist_ok=True)

        params = {
            'main_asset_symbol': main_asset,
            'initial_portfolio_value_usdt': 10000.0,
            'target_weights_normal': { 
                f"{main_asset}_SPOT": 0.5, 
                "USDT": 0.5 # Explicitly including USDT for predictable initial rebalance
            },
            'rebalance_threshold': 0.01, 
            'taker_commission_rate': 0.0,
            'maker_commission_rate': 0.0,
            'use_maker_fees_in_backtest': False,
            'slippage_percentage': 0.0,
            'circuit_breaker_threshold_percent': 0.0, 
            'safe_mode_config': {"enabled": False},
            'min_rebalance_interval_minutes': 0,
            'data_settings': {
                'csv_file_path': '', 
                'signals_csv_path': '', 
                'timestamp_col': "timestamp",
                'ohlc_cols': {"open": "open", "high": "high", "low": "low", "close": "close"},
                'volume_col': "volume",
                'price_col_for_rebalance': "close"
            },
            'logging_level': "INFO", 
            'generate_reports_for_optimizer_trial': False, 
            'report_path_prefix': str(self.temp_reports_dir / "test_run_") 
        }
        if apply_signal_logic_value != "OMIT_KEY":
            params['apply_signal_logic'] = apply_signal_logic_value
        return params

    def test_buy_signal_prevents_spot_sell_logic_enabled(self):
        asset_symbol = "TBUY_ENABLED"
        params = self._get_base_params(main_asset=asset_symbol, apply_signal_logic_value=True)
        spot_key = f"{asset_symbol}_SPOT"
        # params['target_weights_normal'] already set in _get_base_params correctly

        # Market data: SPOT becomes overweight
        ts_initial = pd.Timestamp('2023-01-01T00:00:00Z')
        ts_signal_active = pd.Timestamp('2023-01-01T00:01:00Z')

        market_file = create_dummy_csv(
            self.test_data_dir / f"market_{asset_symbol}_prev_sell.csv", 
            {
                'timestamp': [ts_initial, ts_signal_active],
                'open': [100, 110], 'high': [100, 110], 'low': [100, 110], 'close': [100, 110], 
                'volume': [10,10]
            }
        )
        signal_file = create_dummy_csv(
            self.test_data_dir / f"signal_{asset_symbol}_buy.csv", 
            {'timestamp': [ts_signal_active.isoformat()], 'signal': ['BUY']}
        )
        params['data_settings']['csv_file_path'] = market_file
        params['data_settings']['signals_csv_path'] = signal_file
        
        # Expected behavior:
        # 1. Initial rebalance at ts_initial: Buys SPOT to reach 0.5 weight (50 SPOT if price is 100). This is 1 trade.
        # 2. At ts_signal_active: Price up to 110. SPOT value 50*110 = 5500. USDT 5000. Total 10500.
        #    SPOT weight = 5500/10500 = ~0.5238 (Target 0.5). Overweight.
        #    Normally, would sell SPOT. BUY signal should prevent this.
        
        with self.assertLogs(level='INFO') as log:
            results = run_backtest(params, market_file, is_optimizer_call=False)
        
        self.assertTrue(any(f"Signal BUY for {asset_symbol}: Preventing SELL of {spot_key}" in message for message in log.output),
                        "Log message for BUY signal preventing SPOT sell not found.")
        
        # Check that only the initial rebalance trade occurred
        self.assertEqual(results['total_trades'], 1, "Expected only initial rebalance trades when logic is enabled and signal active.")


    def test_sell_signal_prevents_spot_buy_logic_enabled(self):
        asset_symbol = "TSELL_ENABLED"
        params = self._get_base_params(main_asset=asset_symbol, apply_signal_logic_value=True)
        spot_key = f"{asset_symbol}_SPOT"

        ts_initial = pd.Timestamp('2023-01-01T00:00:00Z')
        ts_signal_active = pd.Timestamp('2023-01-01T00:01:00Z')

        market_file = create_dummy_csv(
            self.test_data_dir / f"market_{asset_symbol}_prev_buy.csv", 
            {
                'timestamp': [ts_initial, ts_signal_active],
                'open': [100, 90], 'high': [100, 90], 'low': [100, 90], 'close': [100, 90], 
                'volume': [10,10]
            }
        )
        signal_file = create_dummy_csv(
            self.test_data_dir / f"signal_{asset_symbol}_sell.csv", 
            {'timestamp': [ts_signal_active.isoformat()], 'signal': ['SELL']}
        )
        params['data_settings']['csv_file_path'] = market_file
        params['data_settings']['signals_csv_path'] = signal_file
        
        with self.assertLogs(level='INFO') as log:
            results = run_backtest(params, market_file, is_optimizer_call=False)

        self.assertTrue(any(f"Signal SELL for {asset_symbol}: Preventing BUY of {spot_key}" in message for message in log.output),
                        "Log message for SELL signal preventing SPOT buy not found.")
        self.assertEqual(results['total_trades'], 1, "Expected only initial rebalance trades when logic is enabled and signal active.")

    def test_buy_signal_allows_spot_sell_logic_disabled(self):
        asset_symbol = "TBUY_DISABLED"
        params = self._get_base_params(main_asset=asset_symbol, apply_signal_logic_value=False)
        spot_key = f"{asset_symbol}_SPOT"

        ts_initial = pd.Timestamp('2023-01-01T00:00:00Z')
        ts_signal_active = pd.Timestamp('2023-01-01T00:01:00Z')
        market_file = create_dummy_csv(
            self.test_data_dir / f"market_{asset_symbol}_allows_sell.csv", 
            {'timestamp': [ts_initial, ts_signal_active], 'open': [100, 110], 'high': [100, 110], 'low': [100, 110], 'close': [100, 110], 'volume': [10,10]}
        )
        signal_file = create_dummy_csv(
            self.test_data_dir / f"signal_{asset_symbol}_buy_allows.csv", 
            {'timestamp': [ts_signal_active.isoformat()], 'signal': ['BUY']}
        )
        params['data_settings']['csv_file_path'] = market_file
        params['data_settings']['signals_csv_path'] = signal_file

        # Expected: Initial rebalance (1 trade) + SPOT sell at ts_signal_active (1 trade) = 2 trades
        results = run_backtest(params, market_file, is_optimizer_call=False)
        self.assertEqual(results['total_trades'], 2, "Expected rebalance sell trade to occur when signal logic is disabled.")

    def test_sell_signal_allows_spot_buy_logic_disabled(self):
        asset_symbol = "TSELL_DISABLED"
        params = self._get_base_params(main_asset=asset_symbol, apply_signal_logic_value=False)
        spot_key = f"{asset_symbol}_SPOT"

        ts_initial = pd.Timestamp('2023-01-01T00:00:00Z')
        ts_signal_active = pd.Timestamp('2023-01-01T00:01:00Z')
        market_file = create_dummy_csv(
            self.test_data_dir / f"market_{asset_symbol}_allows_buy.csv", 
            {'timestamp': [ts_initial, ts_signal_active], 'open': [100, 90], 'high': [100, 90], 'low': [100, 90], 'close': [100, 90], 'volume': [10,10]}
        )
        signal_file = create_dummy_csv(
            self.test_data_dir / f"signal_{asset_symbol}_sell_allows.csv", 
            {'timestamp': [ts_signal_active.isoformat()], 'signal': ['SELL']}
        )
        params['data_settings']['csv_file_path'] = market_file
        params['data_settings']['signals_csv_path'] = signal_file

        # Expected: Initial rebalance (1 trade) + SPOT buy at ts_signal_active (1 trade) = 2 trades
        results = run_backtest(params, market_file, is_optimizer_call=False)
        self.assertEqual(results['total_trades'], 2, "Expected rebalance buy trade to occur when signal logic is disabled.")

    def test_buy_signal_prevents_spot_sell_flag_missing(self):
        asset_symbol = "TBUY_FLAG_MISSING"
        # Pass "OMIT_KEY" (or whatever sentinel you chose) to _get_base_params
        params = self._get_base_params(main_asset=asset_symbol, apply_signal_logic_value="OMIT_KEY") 
        spot_key = f"{asset_symbol}_SPOT"

        ts_initial = pd.Timestamp('2023-01-01T00:00:00Z')
        ts_signal_active = pd.Timestamp('2023-01-01T00:01:00Z')
        market_file = create_dummy_csv(
            self.test_data_dir / f"market_{asset_symbol}_flag_missing.csv", 
            {'timestamp': [ts_initial, ts_signal_active], 'open': [100, 110], 'high': [100, 110], 'low': [100, 110], 'close': [100, 110], 'volume': [10,10]}
        )
        signal_file = create_dummy_csv(
            self.test_data_dir / f"signal_{asset_symbol}_buy_flag_missing.csv", 
            {'timestamp': [ts_signal_active.isoformat()], 'signal': ['BUY']}
        )
        params['data_settings']['csv_file_path'] = market_file
        params['data_settings']['signals_csv_path'] = signal_file

        with self.assertLogs(level='INFO') as log: # Check for both WARNING (missing flag) and INFO (prevention)
            results = run_backtest(params, market_file, is_optimizer_call=False)
        
        self.assertTrue(any("'apply_signal_logic' not found" in message for message in log.output),
                        "Warning for missing 'apply_signal_logic' flag not found.")
        self.assertTrue(any(f"Signal BUY for {asset_symbol}: Preventing SELL of {spot_key}" in message for message in log.output),
                        "Log message for BUY signal preventing SPOT sell not found (flag missing case).")
        self.assertEqual(results['total_trades'], 1, "Expected only initial rebalance trades when flag is missing (defaults to True).")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
