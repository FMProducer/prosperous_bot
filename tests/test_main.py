import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from prosperous_bot.rebalance_backtester import main

@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_data = {
        "backtest_settings": {
            "main_asset_symbol": "BTC",
            "data_settings": {
                "csv_file_path": "data/{main_asset_symbol}USDT_data.csv"
            }
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return str(config_path)

@patch('argparse.ArgumentParser')
@patch('prosperous_bot.rebalance_backtester.run_standalone_backtest')
def test_main_happy_path(mock_run_backtest, mock_argparse, dummy_config_file):
    mock_args = MagicMock()
    mock_args.config_file = dummy_config_file
    mock_args.override = None
    mock_argparse.return_value.parse_args.return_value = mock_args

    main()

    mock_run_backtest.assert_called_once()

@patch('argparse.ArgumentParser')
@patch('prosperous_bot.rebalance_backtester.run_standalone_backtest')
def test_main_override(mock_run_backtest, mock_argparse, dummy_config_file):
    mock_args = MagicMock()
    mock_args.config_file = dummy_config_file
    mock_args.override = '{"main_asset_symbol":"ETH"}'
    mock_argparse.return_value.parse_args.return_value = mock_args

    main()

    mock_run_backtest.assert_called_once()
    args, kwargs = mock_run_backtest.call_args
    assert args[0]["main_asset_symbol"] == "ETH"

@patch('argparse.ArgumentParser')
@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
@patch("builtins.open", new_callable=mock_open)
@patch('json.dump')
@patch('pandas.DataFrame.to_csv')
def test_main_dummy_config_creation(mock_to_csv, mock_json_dump, mock_open, mock_makedirs, mock_exists, mock_argparse):
    mock_args = MagicMock()
    mock_args.config_file = "non_existent_config.json"
    mock_args.override = None
    mock_argparse.return_value.parse_args.return_value = mock_args
    mock_open.side_effect = [FileNotFoundError, MagicMock(), MagicMock()]

    main()

    mock_json_dump.assert_called_once()
    mock_to_csv.assert_called()

@patch('argparse.ArgumentParser')
def test_main_missing_backtest_settings(mock_argparse, tmp_path):
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump({}, f)
    mock_args = MagicMock()
    mock_args.config_file = str(config_path)
    mock_args.override = None
    mock_argparse.return_value.parse_args.return_value = mock_args

    main()

@patch('argparse.ArgumentParser')
def test_main_missing_csv_path(mock_argparse, tmp_path):
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump({"backtest_settings": {"main_asset_symbol": "BTC", "data_settings": {}}}, f)
    mock_args = MagicMock()
    mock_args.config_file = str(config_path)
    mock_args.override = None
    mock_argparse.return_value.parse_args.return_value = mock_args

    main()

@patch('argparse.ArgumentParser')
@patch('prosperous_bot.rebalance_backtester.run_standalone_backtest')
def test_main_json_decode_error_override(mock_run_backtest, mock_argparse, dummy_config_file):
    mock_args = MagicMock()
    mock_args.config_file = dummy_config_file
    mock_args.override = '{"main_asset_symbol":"ETH"' # Invalid JSON
    mock_argparse.return_value.parse_args.return_value = mock_args

    main()

    mock_run_backtest.assert_called_once()

@patch('argparse.ArgumentParser')
def test_main_json_decode_error_config(mock_argparse, tmp_path):
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        f.write("{\"backtest_settings\":{\"main_asset_symbol\":\"BTC\"}") # Invalid JSON
    mock_args = MagicMock()
    mock_args.config_file = str(config_path)
    mock_args.override = None
    mock_argparse.return_value.parse_args.return_value = mock_args

    main()