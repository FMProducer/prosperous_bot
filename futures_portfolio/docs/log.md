(base) PS C:\Python\Prosperous_Bot> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& C:\Python\Prosperous_Bot\.venv\Scripts\Activate.ps1)
(.venv) (base) PS C:\Python\Prosperous_Bot> cd C:\Python\Prosperous_Bot\futures_portfolio
(.venv) (base) PS C:\Python\Prosperous_Bot\futures_portfolio> pytest tests/
================================================================================================================== test session starts ==================================================================================================================
platform win32 -- Python 3.13.2, pytest-8.4.2, pluggy-1.6.0 -- C:\Python\Prosperous_Bot\.venv\Scripts\python.exe
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: C:\Python\Prosperous_Bot\futures_portfolio
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.140.2, asyncio-1.2.0, cov-7.0.0, mock-3.15.1
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 327 items                                                                                                                                                                                                                                      

tests/test_aggregator.py::test_load_config_valid PASSED                                                                                                                                                                                            [  0%]
tests/test_aggregator.py::test_load_config_missing PASSED                                                                                                                                                                                          [  0%]
tests/test_aggregator.py::test_process_telegram_queue PASSED                                                                                                                                                                                       [  0%]
tests/test_aggregator.py::test_generate_swarm_section PASSED                                                                                                                                                                                       [  1%]
tests/test_aggregator.py::test_collect_and_send PASSED                                                                                                                                                                                             [  1%]
tests/test_aggregator.py::test_aggregator_run PASSED                                                                                                                                                                                               [  1%]
tests/test_aggregator.py::test_process_telegram_queue_photo PASSED                                                                                                                                                                                 [  2%]
tests/test_backtest_rebalance.py::test_slippage_simulator PASSED                                                                                                                                                                                   [  2%]
tests/test_backtest_rebalance.py::test_helper_functions PASSED                                                                                                                                                                                     [  2%]
tests/test_backtest_rebalance.py::test_backtest_state PASSED                                                                                                                                                                                       [  3%]
tests/test_backtest_rebalance.py::test_run_backtest_normal_and_guards PASSED                                                                                                                                                                       [  3%]
tests/test_backtest_rebalance.py::test_run_backtest_liquidation_and_trailing_stop PASSED                                                                                                                                                           [  3%]
tests/test_backtest_rebalance.py::test_run_backtest_low_dormant_capital PASSED                                                                                                                                                                     [  3%]
tests/test_backtest_rebalance.py::test_download_live_data PASSED                                                                                                                                                                                   [  4%]
tests/test_backtest_rebalance.py::test_download_live_data_api_error PASSED                                                                                                                                                                         [  4%]
tests/test_backtest_rebalance.py::test_download_live_data_empty PASSED                                                                                                                                                                             [  4%]
tests/test_backtest_rebalance.py::test_cli_execution PASSED                                                                                                                                                                                        [  5%]
tests/test_calculator.py::test_calculator_initialization PASSED                                                                                                                                                                                    [  5%]
tests/test_calculator.py::test_calculate_deviations PASSED                                                                                                                                                                                         [  5%]
tests/test_calculator.py::test_siphoning_reserve_impact PASSED                                                                                                                                                                                     [  6%]
tests/test_calculator.py::test_price_change_impact PASSED                                                                                                                                                                                          [  6%]
tests/test_calculator.py::test_negative_tpv_protection PASSED                                                                                                                                                                                      [  6%]
tests/test_calculator.py::test_ignore_limits_deviation PASSED                                                                                                                                                                                      [  7%]
tests/test_calculator.py::test_limits_deviation PASSED                                                                                                                                                                                             [  7%]
tests/test_calculator.py::test_anti_churn_fuse PASSED                                                                                                                                                                                              [  7%]
tests/test_calculator.py::test_pnl_protection_mode PASSED                                                                                                                                                                                          [  7%]
tests/test_calculator.py::test_force_block_net_move_guard PASSED                                                                                                                                                                                   [  8%]
tests/test_calculator.py::test_edge_cases PASSED                                                                                                                                                                                                   [  8%]
tests/test_calculator.py::test_virt_pnl_paths PASSED                                                                                                                                                                                               [  8%]
tests/test_connector.py::test_get_positions PASSED                                                                                                                                                                                                 [  9%]
tests/test_connector.py::test_get_futures_prices PASSED                                                                                                                                                                                            [  9%]
tests/test_connector.py::test_get_futures_prices_dict_response PASSED                                                                                                                                                                              [  9%]
tests/test_connector.py::test_get_futures_prices_none_tickers PASSED                                                                                                                                                                               [ 10%]
tests/test_connector.py::test_retry_decorator PASSED                                                                                                                                                                                               [ 10%]
tests/test_connector.py::test_retry_decorator_exhausted PASSED                                                                                                                                                                                     [ 10%]
tests/test_connector.py::test_get_margin_ratio PASSED                                                                                                                                                                                              [ 11%]
tests/test_connector.py::test_get_hedge_mode PASSED                                                                                                                                                                                                [ 11%]
tests/test_connector.py::test_get_free_balance PASSED                                                                                                                                                                                              [ 11%]
tests/test_connector.py::test_place_limit_order PASSED                                                                                                                                                                                             [ 11%]
tests/test_connector.py::test_get_spot_prices PASSED                                                                                                                                                                                               [ 12%]
tests/test_connector.py::test_get_futures_klines PASSED                                                                                                                                                                                            [ 12%]
tests/test_connector.py::test_cancel_order PASSED                                                                                                                                                                                                  [ 12%]
tests/test_connector.py::test_place_limit_maker_order PASSED                                                                                                                                                                                       [ 13%]
tests/test_connector.py::test_connector_init_real_mode PASSED                                                                                                                                                                                      [ 13%]
tests/test_connector.py::test_verify_connection_testnet_true PASSED                                                                                                                                                                                [ 13%]
tests/test_connector.py::test_verify_connection_testnet_false PASSED                                                                                                                                                                               [ 14%]
tests/test_connector.py::test_verify_connection_already_initialized PASSED                                                                                                                                                                         [ 14%]
tests/test_connector.py::test_verify_connection_init_binance_api_exception PASSED                                                                                                                                                                  [ 14%]
tests/test_connector.py::test_verify_connection_init_request_exception PASSED                                                                                                                                                                      [ 14%]
tests/test_connector.py::test_verify_connection_ping_binance_api_exception PASSED                                                                                                                                                                  [ 15%]
tests/test_connector.py::test_verify_connection_ping_request_exception PASSED                                                                                                                                                                      [ 15%]
tests/test_connector.py::test_get_position_risk_empty_or_zero PASSED                                                                                                                                                                               [ 15%]
tests/test_connector.py::test_get_position_risk_happy_path PASSED                                                                                                                                                                                  [ 16%]
tests/test_connector.py::test_get_position_risk_edge_cases PASSED                                                                                                                                                                                  [ 16%]
tests/test_connector.py::test_get_order_book PASSED                                                                                                                                                                                                [ 16%]
tests/test_connector.py::test_get_order_status PASSED                                                                                                                                                                                              [ 17%]
tests/test_connector.py::test_get_order_trades PASSED                                                                                                                                                                                              [ 17%]
tests/test_connector.py::test_set_leverage PASSED                                                                                                                                                                                                  [ 17%]
tests/test_connector.py::test_set_margin_type_success PASSED                                                                                                                                                                                       [ 18%]
tests/test_connector.py::test_set_margin_type_already_set PASSED                                                                                                                                                                                   [ 18%]
tests/test_connector.py::test_set_margin_type_other_exception PASSED                                                                                                                                                                               [ 18%]
tests/test_connector.py::test_get_bnb_balance PASSED                                                                                                                                                                                               [ 18%]
tests/test_connector.py::test_get_free_balance_edge_cases PASSED                                                                                                                                                                                   [ 19%]
tests/test_connector.py::test_get_exchange_info PASSED                                                                                                                                                                                             [ 19%]
tests/test_connector.py::test_get_mark_prices_extra PASSED                                                                                                                                                                                         [ 19%]
tests/test_connector.py::test_get_spot_prices_extra PASSED                                                                                                                                                                                         [ 20%]
tests/test_connector.py::test_retry_on_network_error_requests_exceptions PASSED                                                                                                                                                                    [ 20%]
tests/test_connector.py::test_retry_on_network_error_binance_server_error PASSED                                                                                                                                                                   [ 20%]
tests/test_connector.py::test_retry_on_network_error_client_errors PASSED                                                                                                                                                                          [ 21%]
tests/test_connector.py::test_retry_on_network_error_exhausted_server_error PASSED                                                                                                                                                                 [ 21%]
tests/test_connector.py::test_retry_on_network_error_binance_none_status_code PASSED                                                                                                                                                               [ 21%]
tests/test_connector.py::test_retry_on_network_error_aiohttp_client_error PASSED                                                                                                                                                                   [ 22%]
tests/test_connector.py::test_binance_connector_mock PASSED                                                                                                                                                                                        [ 22%]
tests/test_executor.py::test_calculate_order_size PASSED                                                                                                                                                                                           [ 22%]
tests/test_executor.py::test_round_quantity PASSED                                                                                                                                                                                                 [ 22%]
tests/test_executor.py::test_execute_market_order_success PASSED                                                                                                                                                                                   [ 23%]
tests/test_executor.py::test_execute_market_order_rounding_zero PASSED                                                                                                                                                                             [ 23%]
tests/test_executor.py::test_execute_market_order_too_small PASSED                                                                                                                                                                                 [ 23%]
tests/test_executor.py::test_execute_market_order_price_fetch_error PASSED                                                                                                                                                                         [ 24%]
tests/test_executor.py::test_execute_market_order_api_error PASSED                                                                                                                                                                                 [ 24%]
tests/test_executor.py::test_execute_limit_with_fallback_success PASSED                                                                                                                                                                            [ 24%]
tests/test_executor.py::test_execute_limit_with_fallback_too_small PASSED                                                                                                                                                                          [ 25%]
tests/test_executor.py::test_execute_limit_with_fallback_error_then_market PASSED                                                                                                                                                                  [ 25%]
tests/test_executor.py::test_execute_limit_with_fallback_timeout PASSED                                                                                                                                                                            [ 25%]
tests/test_executor.py::test_get_limit_order_params PASSED                                                                                                                                                                                         [ 25%]
tests/test_executor.py::test_execute_actions_surplus_first PASSED                                                                                                                                                                                  [ 26%]
tests/test_executor.py::test_execute_single_action_paper_mode PASSED                                                                                                                                                                               [ 26%]
tests/test_executor.py::test_execute_single_action_real_mode PASSED                                                                                                                                                                                [ 26%]
tests/test_executor.py::test_execute_market_order_polling PASSED                                                                                                                                                                                   [ 27%]
tests/test_executor.py::test_error_returns_contain_pnl_comm PASSED                                                                                                                                                                                 [ 27%]
tests/test_executor.py::test_execute_rebalance PASSED                                                                                                                                                                                              [ 27%]
tests/test_executor.py::test_execute_single_action_virtual PASSED                                                                                                                                                                                  [ 28%]
tests/test_executor.py::test_execute_limit_immediate_fill PASSED                                                                                                                                                                                   [ 28%]
tests/test_executor.py::test_execute_single_action_skipped_dust PASSED                                                                                                                                                                             [ 28%]
tests/test_executor.py::test_execute_single_action_real_limit PASSED                                                                                                                                                                               [ 29%]
tests/test_get_bot_efficiency.py::test_get_bot_efficiency_no_state PASSED                                                                                                                                                                          [ 29%]
tests/test_get_bot_efficiency.py::test_get_bot_efficiency_with_state PASSED                                                                                                                                                                        [ 29%]
tests/test_get_bot_efficiency.py::test_get_bot_efficiency_min_cycles PASSED                                                                                                                                                                        [ 29%]
tests/test_health_check.py::test_check_pm2_success PASSED                                                                                                                                                                                          [ 30%]
tests/test_health_check.py::test_check_pm2_nonzero_return PASSED                                                                                                                                                                                   [ 30%]
tests/test_health_check.py::test_check_pm2_not_found PASSED                                                                                                                                                                                        [ 30%]
tests/test_health_check.py::test_check_pm2_exception PASSED                                                                                                                                                                                        [ 31%]
tests/test_health_check.py::test_check_state_files BTCUSDT content on disk BEFORE calling: {"last_tpv": 10500.0, "last_profit": 500.0, "rebalance_cycles": 42, "trailing_stop_triggered": false, "tpv_ath": 11000.0, "balance": 10000.0}
Returned state files: {'BTCUSDT': {'tpv': 10500.0, 'profit': 500.0, 'cycles': 42, 'trailing_stop': False, 'tpv_ath': 11000.0, 'balance': 10000.0}, 'ETHUSDT': {'error': 'Expecting value: line 1 column 1 (char 0)'}}
PASSED                                                                                                                                                                                          [ 31%]
tests/test_health_check.py::test_check_blacklist PASSED                                                                                                                                                                                            [ 31%]
tests/test_health_check.py::test_check_blacklist_missing PASSED                                                                                                                                                                                    [ 32%]
tests/test_health_check.py::test_check_recent_errors PASSED                                                                                                                                                                                        [ 32%]
tests/test_health_check.py::test_main PASSED                                                                                                                                                                                                       [ 32%]
tests/test_main.py::test_rebalance_loop_siphoning PASSED                                                                                                                                                                                           [ 33%]
tests/test_main.py::test_rebalance_loop_trailing_stop PASSED                                                                                                                                                                                       [ 33%]
tests/test_main.py::test_rebalance_loop_margin_warning PASSED                                                                                                                                                                                      [ 33%]
tests/test_main.py::test_rebalance_loop_margin_critical PASSED                                                                                                                                                                                     [ 33%]
tests/test_main.py::test_clean_slate_protocol_activation PASSED                                                                                                                                                                                    [ 34%]
tests/test_main.py::test_emit_signal_file_naming PASSED                                                                                                                                                                                            [ 34%]
tests/test_main.py::TestMinNotionalExtraction::test_min_notionals_extracted_from_exchange_info PASSED                                                                                                                                              [ 34%]
tests/test_main.py::TestEffectiveMinNotional::test_effective_min_takes_max_with_buffer PASSED                                                                                                                                                      [ 35%]
tests/test_main.py::TestEffectiveMinNotional::test_grassusdt_rebalances_at_3_7pct PASSED                                                                                                                                                           [ 35%]
tests/test_main.py::TestEffectiveMinNotional::test_yfiusdt_blocked_below_5_1pct PASSED                                                                                                                                                             [ 35%]
tests/test_main.py::TestEffectiveMinNotional::test_fallback_to_config_when_exchange_info_missing PASSED                                                                                                                                            [ 36%]
tests/test_main.py::TestEffectiveMinNotional::test_missing_min_notional_in_config_raises PASSED                                                                                                                                                    [ 36%]
tests/test_main.py::test_handle_liquidation_recovery_no_config_write PASSED                                                                                                                                                                        [ 36%]
tests/test_main.py::test_handle_liquidation_guard_no_config_write PASSED                                                                                                                                                                           [ 37%]
tests/test_main_coverage.py::test_coverage_trailing_stop_timeout_closes_positions PASSED                                                                                                                                                           [ 37%]
tests/test_main_coverage.py::test_coverage_trailing_stop_violation_start PASSED                                                                                                                                                                    [ 37%]
tests/test_main_coverage.py::test_coverage_emergency_stop_blacklist PASSED                                                                                                                                                                         [ 37%]
tests/test_main_coverage.py::test_coverage_emergency_stop_probation PASSED                                                                                                                                                                         [ 38%]
tests/test_main_coverage.py::test_coverage_liquidation_guard_detects_missing_long PASSED                                                                                                                                                           [ 38%]
tests/test_main_coverage.py::test_coverage_virtual_order_processing PASSED                                                                                                                                                                         [ 38%]
tests/test_main_coverage.py::test_coverage_paper_stop_closes_positions PASSED                                                                                                                                                                      [ 39%]
tests/test_main_coverage.py::test_coverage_real_stop_closes_positions PASSED                                                                                                                                                                       [ 39%]
tests/test_main_coverage.py::test_coverage_margin_warning_real PASSED                                                                                                                                                                              [ 39%]
tests/test_main_coverage.py::test_coverage_liquidity_guard PASSED                                                                                                                                                                                  [ 40%]
tests/test_main_coverage.py::test_coverage_ticker_thresholds_dict PASSED                                                                                                                                                                           [ 40%]
tests/test_main_coverage.py::test_coverage_ticker_thresholds_float PASSED                                                                                                                                                                          [ 40%]
tests/test_main_coverage.py::test_coverage_siphoning_logic PASSED                                                                                                                                                                                  [ 40%]
tests/test_main_coverage.py::test_coverage_config_reload_detection PASSED                                                                                                                                                                          [ 41%]
tests/test_main_coverage.py::test_coverage_liquidation_guard_missing_long PASSED                                                                                                                                                                   [ 41%]
tests/test_main_coverage.py::test_coverage_blacklist_rebase_loss PASSED                                                                                                                                                                            [ 41%]
tests/test_main_coverage.py::test_coverage_paper_cross_margin_check PASSED                                                                                                                                                                         [ 42%]
tests/test_main_coverage.py::test_coverage_trailing_stop_closure_paper_positions PASSED                                                                                                                                                            [ 42%]
tests/test_main_coverage.py::test_coverage_emergency_stop_full PASSED                                                                                                                                                                              [ 42%]
tests/test_main_coverage.py::test_coverage_emergency_stop_close_only PASSED                                                                                                                                                                        [ 43%]
tests/test_main_coverage.py::test_coverage_handle_liquidation_recovery_closes_positions PASSED                                                                                                                                                     [ 43%]
tests/test_main_new.py::test_rebalance_loop_siphoning PASSED                                                                                                                                                                                       [ 43%]
tests/test_main_new.py::test_rebalance_loop_trailing_stop PASSED                                                                                                                                                                                   [ 44%]
tests/test_main_new.py::test_rebalance_loop_margin_warning PASSED                                                                                                                                                                                  [ 44%]
tests/test_main_new.py::test_rebalance_loop_margin_critical PASSED                                                                                                                                                                                 [ 44%]
tests/test_main_new.py::test_clean_slate_protocol_activation PASSED                                                                                                                                                                                [ 44%]
tests/test_main_new.py::test_emit_signal_file_naming PASSED                                                                                                                                                                                        [ 45%]
tests/test_main_new.py::test_self_kill_pm2_flow PASSED                                                                                                                                                                                             [ 45%]
tests/test_main_new.py::test_update_final_metrics_for_exit_flow PASSED                                                                                                                                                                             [ 45%]
tests/test_main_new.py::test_handle_liquidation_recovery_flow PASSED                                                                                                                                                                               [ 46%]
tests/test_main_new.py::test_rebalance_loop_zombie_on_startup PASSED                                                                                                                                                                               [ 46%]
tests/test_main_new.py::test_handle_liquidation_guard_paper_critical PASSED                                                                                                                                                                        [ 46%]
tests/test_main_new.py::test_handle_liquidation_guard_real_critical PASSED                                                                                                                                                                         [ 47%]
tests/test_main_new.py::test_handle_liquidation_guard_warning PASSED                                                                                                                                                                               [ 47%]
tests/test_notifier.py::test_notifier_init PASSED                                                                                                                                                                                                  [ 47%]
tests/test_notifier.py::test_notifier_disabled PASSED                                                                                                                                                                                              [ 48%]
tests/test_notifier.py::test_load_config_exception PASSED                                                                                                                                                                                          [ 48%]
tests/test_notifier.py::test_session_creation PASSED                                                                                                                                                                                               [ 48%]
tests/test_notifier.py::test_queue_message_text PASSED                                                                                                                                                                                             [ 48%]
tests/test_notifier.py::test_queue_message_exception PASSED                                                                                                                                                                                        [ 49%]
tests/test_notifier.py::test_send_message_direct_success PASSED                                                                                                                                                                                    [ 49%]
tests/test_notifier.py::test_send_message_direct_429_short_retry PASSED                                                                                                                                                                            [ 49%]
tests/test_notifier.py::test_send_message_direct_429_long_retry_skip PASSED                                                                                                                                                                        [ 50%]
tests/test_notifier.py::test_send_message_direct_api_error PASSED                                                                                                                                                                                  [ 50%]
tests/test_notifier.py::test_send_message_connection_error PASSED                                                                                                                                                                                  [ 50%]
tests/test_notifier.py::test_send_photo_missing_path PASSED                                                                                                                                                                                        [ 51%]
tests/test_notifier.py::test_send_photo_queue PASSED                                                                                                                                                                                               [ 51%]
tests/test_notifier.py::test_send_photo_direct_success PASSED                                                                                                                                                                                      [ 51%]
tests/test_notifier.py::test_send_photo_direct_429 PASSED                                                                                                                                                                                          [ 51%]
tests/test_notifier.py::test_send_photo_direct_429_long PASSED                                                                                                                                                                                     [ 52%]
tests/test_notifier.py::test_send_photo_direct_api_error PASSED                                                                                                                                                                                    [ 52%]
tests/test_notifier.py::test_send_photo_exception PASSED                                                                                                                                                                                           [ 52%]
tests/test_notifier.py::test_send_alert PASSED                                                                                                                                                                                                     [ 53%]
tests/test_notifier.py::test_send_status PASSED                                                                                                                                                                                                    [ 53%]
tests/test_rank_tickers.py::test_fetch_success PASSED                                                                                                                                                                                              [ 53%]
tests/test_rank_tickers.py::test_fetch_klines PASSED                                                                                                                                                                                               [ 54%]
tests/test_rank_tickers.py::test_fetch_rate_limit PASSED                                                                                                                                                                                           [ 54%]
tests/test_rank_tickers.py::test_fetch_non_200 PASSED                                                                                                                                                                                              [ 54%]
tests/test_rank_tickers.py::test_fetch_exception PASSED                                                                                                                                                                                            [ 55%]
tests/test_rank_tickers.py::test_fetch_klines_too_few PASSED                                                                                                                                                                                       [ 55%]
tests/test_rank_tickers.py::test_fetch_klines_empty PASSED                                                                                                                                                                                         [ 55%]
tests/test_rank_tickers.py::test_get_top_tickers PASSED                                                                                                                                                                                            [ 55%]
tests/test_rank_tickers.py::test_get_top_tickers_fetch_fail PASSED                                                                                                                                                                                 [ 56%]
tests/test_rank_tickers.py::test_get_top_tickers_empty_dfs PASSED                                                                                                                                                                                  [ 56%]
tests/test_rank_tickers.py::test_get_top_tickers_blacklist_filter PASSED                                                                                                                                                                           [ 56%]
tests/test_rank_tickers.py::test_get_top_tickers_whitelist_filter PASSED                                                                                                                                                                           [ 57%]
tests/test_rank_tickers.py::test_get_top_tickers_low_volume_filter PASSED                                                                                                                                                                          [ 57%]
tests/test_rank_tickers.py::test_get_top_tickers_non_ascii PASSED                                                                                                                                                                                  [ 57%]
tests/test_rank_tickers.py::test_ranker_init PASSED                                                                                                                                                                                                [ 58%]
tests/test_rank_tickers.py::test_ranker_calculate_metrics PASSED                                                                                                                                                                                   [ 58%]
tests/test_rank_tickers.py::test_ranker_rank_by_momentum PASSED                                                                                                                                                                                    [ 58%]
tests/test_rank_tickers.py::test_run_ranker_task PASSED                                                                                                                                                                                            [ 59%]
tests/test_rank_tickers.py::test_retry_on_network_error_exhausted PASSED                                                                                                                                                                           [ 59%]
tests/test_rank_tickers.py::test_retry_on_network_error_success PASSED                                                                                                                                                                             [ 59%]
tests/test_rank_tickers.py::test_main_with_config PASSED                                                                                                                                                                                           [ 59%]
tests/test_rank_tickers.py::test_main_quiet PASSED                                                                                                                                                                                                 [ 60%]
tests/test_rank_tickers.py::test_main_not_quiet 
=============================================================================================================================
SYMBOL          | CYCLES   | NET MOVE%    | MAX SPURT%   | TREND EFF%   | FUNDING%
-----------------------------------------------------------------------------------------------------------------------------
BTCUSDT         | 20       | 5.00         | 10.00        | 2.00         | 0.0100
=============================================================================================================================
PASSED                                                                                                                                                                                             [ 60%]
tests/test_rank_tickers.py::test_main_empty PASSED                                                                                                                                                                                                 [ 60%]
tests/test_rebalance_logic_v378.py::test_rebalance_logic_v378 
--- CASE 1: Perfect Balance ---
DEBUG: L:29.0% S:36.0% V:35.0% C:0.0%
--- CASE 2: LONG Surplus (Sell to Cash) ---
DEBUG Case 2: L:58.0% S:0.0% V:42.0% C:0.0%
DEBUG Actions Case 2: [{'key': 'BASE_LONG', 'type': 'ORDER', 'symbol': 'BTCUSDT_LONG', 'base_symbol': 'BTCUSDT', 'position_side': 'LONG', 'diff_usdt': -145.0, 'diff_equity': -29.0, 'leverage': 5.0, 'is_reduction': True, 'priority': 0}, {'key': 'VIRTUAL', 'type': 'VIRTUAL_ORDER', 'symbol': 'VIRTUAL', 'base_symbol': 'VIRTUAL', 'position_side': 'BOTH', 'diff_usdt': -7.0, 'diff_equity': -7.0, 'leverage': 1.0, 'is_reduction': True, 'priority': 0}, {'key': 'BASE_SHORT', 'type': 'ORDER', 'symbol': 'BTCUSDT_SHORT', 'base_symbol': 'BTCUSDT', 'position_side': 'SHORT', 'diff_usdt': 180.0, 'diff_equity': 36.0, 'leverage': 5.0, 'is_reduction': False, 'priority': 2}]
--- CASE 3: Cash Guard (Insufficient funds for Short) ---
--- CASE 4: Priority BUY (Virtual gets cash first) ---
DEBUG Case 4 Shares: L:41.43% S:28.57% V:28.57% C:1.43%
DEBUG Case 4 Actions: [{'key': 'VIRTUAL', 'type': 'VIRTUAL_ORDER', 'symbol': 'VIRTUAL', 'base_symbol': 'VIRTUAL', 'position_side': 'BOTH', 'diff_usdt': 7.0, 'diff_equity': 15.0, 'leverage': 1.0, 'is_reduction': False, 'priority': 2}]
--- CASE 5: Heartbeat Summation ---

✅ ALL TESTS PASSED!
PASSED                                                                                                                                                                               [ 61%]
tests/test_rebalance_v37.py::test_portfolio_convergence_and_churn 
--- Ребалансировочный Тест v3.7.0 ---
[Price: 26.5] Rebalancing BASE_LONG | Action: -46.545 USDT
[Price: 26.5] Rebalancing BASE_SHORT | Action: 46.545 USDT
[Price: 24.0] Rebalancing BASE_SHORT | Action: -33.48 USDT
[Price: 24.0] Rebalancing BASE_LONG | Action: 31.03 USDT
[Price: 28.0] Rebalancing BASE_LONG | Action: -93.09 USDT
[Price: 28.0] Rebalancing BASE_SHORT | Action: 93.09 USDT
[Price: 22.0] Rebalancing BASE_SHORT | Action: -100.44 USDT
[Price: 22.0] Rebalancing BASE_LONG | Action: 93.09 USDT

Итоговая статистика ребалансировок: {'BASE_LONG': 4, 'BASE_SHORT': 4, 'VIRTUAL': 0}
PASSED                                                                                                                                                                           [ 61%]
tests/test_send_monitor_report.py::test_monitor_no_state_files {
  "ok": true
}
PASSED                                                                                                                                                                              [ 61%]
tests/test_send_monitor_report.py::test_monitor_with_paper_and_real {
  "ok": true
}
PASSED                                                                                                                                                                         [ 62%]
tests/test_send_monitor_report.py::test_monitor_guard_active {
  "ok": true
}
PASSED                                                                                                                                                                                [ 62%]
tests/test_send_monitor_report.py::test_monitor_old_heartbeat {
  "ok": true
}
PASSED                                                                                                                                                                               [ 62%]
tests/test_send_monitor_report.py::test_monitor_sends_request {
  "ok": true
}
PASSED                                                                                                                                                                               [ 62%]
tests/test_send_monitor_report.py::test_monitor_no_alerts {
  "ok": true
}
PASSED                                                                                                                                                                                   [ 63%]
tests/test_storage.py::test_safe_load_json_sync_file_exists PASSED                                                                                                                                                                                 [ 63%]
tests/test_storage.py::test_safe_load_json_sync_file_missing PASSED                                                                                                                                                                                [ 63%]
tests/test_storage.py::test_safe_load_json_sync_empty_file PASSED                                                                                                                                                                                  [ 64%]
tests/test_storage.py::test_safe_load_json_sync_corrupted_json PASSED                                                                                                                                                                              [ 64%]
tests/test_storage.py::test_safe_load_json_async PASSED                                                                                                                                                                                            [ 64%]
tests/test_storage.py::test_safe_save_json_sync_atomic PASSED                                                                                                                                                                                      [ 65%]
tests/test_storage.py::test_safe_save_json_sync_permission_error PASSED                                                                                                                                                                            [ 65%]
tests/test_storage.py::test_safe_save_json_async PASSED                                                                                                                                                                                            [ 65%]
tests/test_storage.py::test_safe_load_json_retries_and_timeout PASSED                                                                                                                                                                              [ 66%]
tests/test_storage.py::test_safe_save_json_sync_remove_error PASSED                                                                                                                                                                                [ 66%]
tests/test_supervisor.py::test_calculate_bot_score 2026-07-14 16:48:36,590 INFO: 🛡️ Scored BTCUSDT: REAL bot in drawdown. LOCKED IN COMBAT (Score: INF).
2026-07-14 16:48:36,590 INFO: ❌ Scored BTCUSDT: Rejected (Cycles 5 < 10). (Score: -INF)
2026-07-14 16:48:36,590 INFO: ❌ Scored BTCUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:48:36,590 INFO: ❌ Scored BTCUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:48:36,590 INFO: ⚖️ Scored BTCUSDT: Net:100.00, Cyc:20, Score:15.2226
2026-07-14 16:48:36,591 INFO: ⚖️ Scored BTCUSDT: Net:100.00, Cyc:20, Score:18.2671
PASSED                                                                                                                                                                                          [ 66%]
tests/test_supervisor.py::test_calc_rotation_score PASSED                                                                                                                                                                                          [ 66%]
tests/test_supervisor.py::test_selective_merge_incubator 2026-07-14 16:48:36,593 INFO: 🚀 First-start: launching 4 bots from scanner: ['A', 'B', 'C', 'D']
2026-07-14 16:48:36,594 INFO: 🔄 Rotation: keep=2, add=2, remove=2
2026-07-14 16:48:36,594 INFO: 🔄   removing (worst by score): ['C', 'D']
2026-07-14 16:48:36,594 INFO: 🔄   adding (best from scanner): ['E', 'F']
2026-07-14 16:48:36,594 INFO: 📋 Final incubator (4 bots): ['A', 'B', 'E', 'F']
PASSED                                                                                                                                                                                    [ 67%]
tests/test_supervisor.py::test_get_bot_efficiency PASSED                                                                                                                                                                                           [ 67%]
tests/test_supervisor.py::test_reset_bot_state_files_real 2026-07-14 16:48:36,599 INFO: 💾 Archived state for BTCUSDT: real_state_BTCUSDT.json -> archive_BTCUSDT_1784040516_real_state_BTCUSDT.json
2026-07-14 16:48:36,600 INFO: 🗑️ Deleted stale state file: real_state_BTCUSDT.json
2026-07-14 16:48:36,600 INFO: 💾 Archived state for BTCUSDT: shadow_state_BTCUSDT.json -> archive_BTCUSDT_1784040516_shadow_state_BTCUSDT.json
2026-07-14 16:48:36,600 INFO: 🗑️ Deleted stale state file: shadow_state_BTCUSDT.json
2026-07-14 16:48:36,601 INFO: ✨ Reset real state files for BTCUSDT to clean initial values (capital: 200.0).
PASSED                                                                                                                                                                                   [ 67%]
tests/test_supervisor.py::test_reset_bot_state_files_paper 2026-07-14 16:48:36,604 INFO: 💾 Archived paper state for BTCUSDT: paper_state_BTCUSDT.json -> archive_BTCUSDT_1784040516_paper_state_BTCUSDT.json
2026-07-14 16:48:36,604 INFO: 🗑️ Deleted stale state file: paper_state_BTCUSDT.json
2026-07-14 16:48:36,604 INFO: 💾 Archived paper state for BTCUSDT: paper_shadow_BTCUSDT.json -> archive_BTCUSDT_1784040516_paper_shadow_BTCUSDT.json
2026-07-14 16:48:36,604 INFO: 🗑️ Deleted stale state file: paper_shadow_BTCUSDT.json
2026-07-14 16:48:36,605 INFO: ✨ Reset paper state files for BTCUSDT to clean initial values (capital: 150.0).
PASSED                                                                                                                                                                                  [ 68%]
tests/test_supervisor.py::test_enforce_swarm_consistency 2026-07-14 16:48:36,608 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
2026-07-14 16:48:36,608 WARNING: ⚠️ Unauthorized position found for ETHUSDT (not in live_swarm or real_whitelist). Scheduled for liquidation.
2026-07-14 16:48:36,608 INFO: 🧹 Closing all positions for unauthorized ticker: ETHUSDT
2026-07-14 16:48:36,609 INFO: ✅ Positions for ETHUSDT closed successfully.
PASSED                                                                                                                                                                                    [ 68%]
tests/test_supervisor.py::test_ensure_real_bots_alive 2026-07-14 16:48:36,611 WARNING: 🔄 Real bot ETHUSDT is in live_swarm but not running in PM2. Restarting...
2026-07-14 16:48:36,612 INFO: ✅ Restarted real bot ETHUSDT
PASSED                                                                                                                                                                                       [ 68%]
tests/test_supervisor.py::test_manage_swarm_toxic_flow 2026-07-14 16:48:36,616 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:36,616 CRITICAL: 🚫 STOP signal: TRXUSDT (paper) → toxic_blacklist_paper + black_list (expires Tue Jul 14 17:17:24 2026)
2026-07-14 16:48:36,617 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:36,617 INFO: ⏳ TRXUSDT is in Toxic Quarantine. Skipping.
2026-07-14 16:48:36,617 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['BTCUSDT']
2026-07-14 16:48:36,617 INFO: 🔄 Rotation: keep=1, add=0, remove=0
2026-07-14 16:48:36,618 INFO: 📋 Final incubator (1 bots): ['BTCUSDT']
2026-07-14 16:48:36,618 INFO: 📋 Incubator ready: ['BTCUSDT']
2026-07-14 16:48:36,618 INFO: 🚫 Skipping BTCUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:36,618 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:36,618 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:48:36,619 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:48:38,121 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:48:38,122 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                                      [ 69%]
tests/test_supervisor.py::test_isolated_blacklists_signal_processing 2026-07-14 16:48:38,126 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:38,127 CRITICAL: 🚫 STOP signal: BTCUSDT (paper) → toxic_blacklist_paper + black_list (expires Tue Jul 14 17:17:26 2026)
2026-07-14 16:48:38,128 INFO: ✅ EXIT signal: ETHUSDT (real) → probation_real (expires Tue Jul 14 17:47:40 2026)
2026-07-14 16:48:38,128 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:38,128 INFO: ⏳ BTCUSDT is in Toxic Quarantine. Skipping.
2026-07-14 16:48:38,128 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['ETHUSDT']
2026-07-14 16:48:38,128 INFO: 🔄 Rotation: keep=1, add=0, remove=0
2026-07-14 16:48:38,128 INFO: 📋 Final incubator (1 bots): ['ETHUSDT']
2026-07-14 16:48:38,129 INFO: 📋 Incubator ready: ['ETHUSDT']
2026-07-14 16:48:38,129 INFO: 🚫 Skipping ETHUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:38,129 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:40,117 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): ETHUSDT
2026-07-14 16:48:40,364 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                        [ 69%]
tests/test_supervisor.py::test_get_running_bots_info PASSED                                                                                                                                                                                        [ 69%]
tests/test_supervisor.py::test_stop_bot PASSED                                                                                                                                                                                                     [ 70%]
tests/test_supervisor.py::test_start_bot PASSED                                                                                                                                                                                                    [ 70%]
tests/test_supervisor.py::test_rotation_guard_immature_unprofitable_protected 2026-07-14 16:48:40,376 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['B']
2026-07-14 16:48:40,376 INFO: 🔄 Rotation: keep=2, add=2, remove=2
2026-07-14 16:48:40,376 INFO: 🔄   removing (worst by score): ['C', 'D']
2026-07-14 16:48:40,376 INFO: 🔄   adding (best from scanner): ['E', 'F']
2026-07-14 16:48:40,376 INFO: 📋 Final incubator (4 bots): ['A', 'B', 'E', 'F']
PASSED                                                                                                                                                               [ 70%]
tests/test_supervisor.py::test_rotation_guard_immature_profitable_protected 2026-07-14 16:48:40,379 INFO: 🔄 Rotation: keep=1, add=2, remove=2
2026-07-14 16:48:40,379 INFO: 🔄   removing (worst by score): ['B', 'C']
2026-07-14 16:48:40,379 INFO: 🔄   adding (best from scanner): ['D', 'E']
2026-07-14 16:48:40,379 INFO: 📋 Final incubator (3 bots): ['A', 'D', 'E']
PASSED                                                                                                                                                                 [ 70%]
tests/test_supervisor.py::test_rotation_guard_mature_unprofitable_replaced 2026-07-14 16:48:40,381 INFO: 🔄 Rotation: keep=1, add=2, remove=2
2026-07-14 16:48:40,381 INFO: 🔄   removing (worst by score): ['B', 'C']
2026-07-14 16:48:40,381 INFO: 🔄   adding (best from scanner): ['D', 'E']
2026-07-14 16:48:40,381 INFO: 📋 Final incubator (3 bots): ['A', 'D', 'E']
PASSED                                                                                                                                                                  [ 71%]
tests/test_supervisor.py::test_rotation_guard_default_min_cycles_for_rotation 2026-07-14 16:48:40,383 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['A']
2026-07-14 16:48:40,384 INFO: 🔄 Rotation: keep=1, add=2, remove=2
2026-07-14 16:48:40,384 INFO: 🔄   removing (worst by score): ['B', 'C']
2026-07-14 16:48:40,384 INFO: 🔄   adding (best from scanner): ['D', 'E']
2026-07-14 16:48:40,384 INFO: 📋 Final incubator (3 bots): ['A', 'D', 'E']
PASSED                                                                                                                                                               [ 71%]
tests/test_supervisor.py::test_b1_stop_bot_called_before_ts_flag_reset 2026-07-14 16:48:40,387 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
2026-07-14 16:48:40,387 WARNING: ⚠️ HEAL REJECTED: GRASSUSDT was stopped by Trailing Stop. Scheduling position liquidation.
2026-07-14 16:48:40,387 INFO: 🔄 TS flags reset in state file for GRASSUSDT
2026-07-14 16:48:40,388 INFO: 🧹 Closing all positions for unauthorized ticker: GRASSUSDT
2026-07-14 16:48:40,388 INFO: ✅ Positions for GRASSUSDT closed successfully.
PASSED                                                                                                                                                                      [ 71%]
tests/test_supervisor.py::test_b1_ts_flag_reset_after_stop_bot 2026-07-14 16:48:40,391 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
2026-07-14 16:48:40,391 WARNING: ⚠️ HEAL REJECTED: VVVUSDT was stopped by Trailing Stop. Scheduling position liquidation.
2026-07-14 16:48:40,392 INFO: 🔄 TS flags reset in state file for VVVUSDT
2026-07-14 16:48:40,392 INFO: 🧹 Closing all positions for unauthorized ticker: VVVUSDT
2026-07-14 16:48:40,392 INFO: ✅ Positions for VVVUSDT closed successfully.
PASSED                                                                                                                                                                              [ 72%]
tests/test_supervisor.py::test_signal_stop_adds_toxic_and_blacklist PASSED                                                                                                                                                                         [ 72%]
tests/test_supervisor.py::test_signal_exit_adds_probation_not_blacklist PASSED                                                                                                                                                                     [ 72%]
tests/test_supervisor.py::test_signal_removes_from_live_swarm PASSED                                                                                                                                                                               [ 73%]
tests/test_supervisor.py::test_signal_unknown_type_no_config_change PASSED                                                                                                                                                                         [ 73%]
tests/test_supervisor.py::test_blacklist_set_o1_lookup PASSED                                                                                                                                                                                      [ 73%]
tests/test_supervisor.py::test_probation_blocks_candidate PASSED                                                                                                                                                                                   [ 74%]
tests/test_supervisor.py::test_probation_expired_not_blocks PASSED                                                                                                                                                                                 [ 74%]
tests/test_supervisor.py::test_flag_unlink_is_async PASSED                                                                                                                                                                                         [ 74%]
tests/test_supervisor.py::test_integration_signal_stop_toxic_blacklist 2026-07-14 16:48:40,430 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:40,433 INFO: 🗑️ BTCUSDT removed from live_swarm (signal: stop)
2026-07-14 16:48:40,433 CRITICAL: 🚫 STOP signal: BTCUSDT (real) → toxic_blacklist_real + black_list (expires Tue Jul 14 17:17:28 2026)
2026-07-14 16:48:40,434 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:40,434 INFO: 📋 Incubator ready: ['X']
2026-07-14 16:48:40,435 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:40,435 INFO: 🚫 Skipping X: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:40,435 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:41,940 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): X
2026-07-14 16:48:41,941 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:41,942 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                      [ 74%]
tests/test_supervisor.py::test_integration_signal_exit_probation 2026-07-14 16:48:41,950 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:41,952 INFO: 🗑️ ZECUSDT removed from live_swarm (signal: exit)
2026-07-14 16:48:41,953 INFO: ✅ EXIT signal: ZECUSDT (paper) → probation_paper (expires Tue Jul 14 17:47:44 2026)
2026-07-14 16:48:41,953 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:41,953 INFO: 📋 Incubator ready: ['X']
2026-07-14 16:48:41,953 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:41,954 INFO: 🚫 Skipping X: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:41,954 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:43,460 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): X
2026-07-14 16:48:43,461 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:43,462 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                            [ 75%]
tests/test_supervisor.py::test_integration_amnesty_clears_ts 2026-07-14 16:48:43,469 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:43,480 INFO: ✨ Amnesty granted for PAPER ALICEUSDT. TS flags cleared.
2026-07-14 16:48:43,481 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:43,481 INFO: 📋 Incubator ready: ['X']
2026-07-14 16:48:43,481 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:43,481 INFO: 🚫 Skipping X: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:43,482 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:44,990 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): X
2026-07-14 16:48:44,991 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:44,991 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                                [ 75%]
tests/test_supervisor.py::test_integration_reaper_zombie 2026-07-14 16:48:45,003 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:45,007 INFO: 🧹 Reaper Guard: Deleting zombie PM2 process for PAPER ALICEUSDT (TS Triggered)
2026-07-14 16:48:45,008 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:45,008 INFO: 📋 Incubator ready: ['X']
2026-07-14 16:48:45,008 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:45,008 INFO: 🚫 Skipping X: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:45,008 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:46,524 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): X
2026-07-14 16:48:46,525 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:46,525 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                                    [ 75%]
tests/test_supervisor.py::test_integration_ready_pool_scoring 2026-07-14 16:48:46,532 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:46,535 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:46,535 INFO: 📋 Incubator ready: ['GOODUSDT']
2026-07-14 16:48:46,535 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:46,535 INFO: ⚖️ Scored GOODUSDT: Net:50.00, Cyc:20, Score:7.6113
2026-07-14 16:48:46,536 INFO: ✅ Filling empty REAL slot with GOODUSDT
2026-07-14 16:48:46,536 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:48,045 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): GOODUSDT
2026-07-14 16:48:48,046 INFO: 🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): GOODUSDT
2026-07-14 16:48:48,047 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:48,047 INFO: Cycle Complete. REAL Swarm: ['GOODUSDT']
PASSED                                                                                                                                                                               [ 76%]
tests/test_supervisor.py::test_integration_real_rotation_fills_slots 2026-07-14 16:48:48,055 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:48,057 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:48,058 INFO: 📋 Incubator ready: ['GOODUSDT']
2026-07-14 16:48:48,058 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:48,058 INFO: ⚖️ Scored GOODUSDT: Net:80.00, Cyc:20, Score:12.1781
2026-07-14 16:48:48,058 INFO: ✅ Filling empty REAL slot with GOODUSDT
2026-07-14 16:48:48,058 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:49,560 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): GOODUSDT
2026-07-14 16:48:49,561 INFO: 🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): GOODUSDT
2026-07-14 16:48:49,562 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:49,562 INFO: Cycle Complete. REAL Swarm: ['GOODUSDT']
PASSED                                                                                                                                                                        [ 76%]
tests/test_supervisor.py::test_integration_replaces_unprofitable 2026-07-14 16:48:49,570 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:49,580 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:49,580 INFO: 📋 Incubator ready: ['GOODUSDT']
2026-07-14 16:48:49,580 INFO: 📊 Fetching missing performance data for 2 candidates...
2026-07-14 16:48:49,581 INFO: ❌ Scored BADUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:48:49,581 INFO: ⚖️ Scored GOODUSDT: Net:80.00, Cyc:20, Score:12.1781
2026-07-14 16:48:49,581 INFO: 🧐 Evaluating REAL BADUSDT [Score: -inf, Profit: $0.00, Age: 168.00h] vs Candidate GOODUSDT [Score: 12.1781]
2026-07-14 16:48:49,581 INFO: ♻️ Substitution Triggered: Replacing BADUSDT with GOODUSDT (Score delta inf > Cushion 0.25)
2026-07-14 16:48:49,582 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:49,582 INFO: 🛑 Stopping Combat REAL process: BADUSDT (Rolling back to pure paper tracking)
2026-07-14 16:48:51,098 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): GOODUSDT
2026-07-14 16:48:51,099 INFO: 🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): GOODUSDT
2026-07-14 16:48:51,105 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:51,105 INFO: Cycle Complete. REAL Swarm: ['GOODUSDT']
PASSED                                                                                                                                                                            [ 76%]
tests/test_supervisor.py::test_integration_stops_out_of_incubator 2026-07-14 16:48:51,113 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:51,116 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:51,116 INFO: 📋 Incubator ready: ['BTCUSDT']
2026-07-14 16:48:51,116 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:51,117 INFO: 🚫 Skipping BTCUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:51,117 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:51,117 INFO: 🛑 Stopping Incubator (Out of Scanner): OLDUSDT
2026-07-14 16:48:52,621 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:48:52,622 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:52,622 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                           [ 77%]
tests/test_supervisor.py::test_integration_probation_blocks 2026-07-14 16:48:52,631 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:52,643 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:52,643 INFO: 📋 Incubator ready: ['BLOCKEDUSDT']
2026-07-14 16:48:52,643 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:52,643 INFO: ⏳ Skipping BLOCKEDUSDT: In probation_paper until Tue Jul 14 17:48:52 2026.
2026-07-14 16:48:52,644 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:54,146 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BLOCKEDUSDT
2026-07-14 16:48:54,146 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:54,147 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                                 [ 77%]
tests/test_supervisor.py::test_integration_drawdown_protection 2026-07-14 16:48:54,156 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:54,159 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:54,159 INFO: 📋 Incubator ready: ['DRAWDOWNUSDT']
2026-07-14 16:48:54,160 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:54,160 INFO: 🛡️ Scored DRAWDOWNUSDT: REAL bot in drawdown. LOCKED IN COMBAT (Score: INF).
2026-07-14 16:48:54,160 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:55,666 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): DRAWDOWNUSDT
2026-07-14 16:48:55,667 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:55,668 INFO: Cycle Complete. REAL Swarm: ['DRAWDOWNUSDT']
PASSED                                                                                                                                                                              [ 77%]
tests/test_supervisor.py::test_integration_toxic_scanner_mark 2026-07-14 16:48:55,675 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:55,677 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:55,677 INFO: 🚫 TOXICUSDT marked toxic by scanner. Blacklisted (PAPER) until Tue Jul 14 17:17:43 2026
2026-07-14 16:48:55,677 INFO: ⏳ TOXICUSDT is in Toxic Quarantine. Skipping.
2026-07-14 16:48:55,678 INFO: 📋 Incubator ready: ['TOXICUSDT']
2026-07-14 16:48:55,678 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:55,678 INFO: 🚫 Skipping TOXICUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:55,678 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:57,192 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): TOXICUSDT
2026-07-14 16:48:57,194 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:57,194 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                               [ 77%]
tests/test_supervisor.py::test_integration_scanner_empty_aborts 2026-07-14 16:48:57,202 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:57,205 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:57,205 ERROR: ❌ Scanner returned nothing. Aborting cycle to prevent config wipe.
PASSED                                                                                                                                                                             [ 78%]
tests/test_supervisor.py::test_integration_expired_probation_pruned 2026-07-14 16:48:57,214 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:57,223 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:57,223 INFO: 📋 Incubator ready: ['X']
2026-07-14 16:48:57,223 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:48:57,224 INFO: 🚫 Skipping X: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:57,224 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:48:58,736 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): X
2026-07-14 16:48:58,737 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:48:58,737 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                         [ 78%]
tests/test_supervisor.py::test_integration_ticks_fallback 2026-07-14 16:48:58,745 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:48:58,748 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:48:58,748 INFO: 📋 Incubator ready: ['BTCUSDT', 'ETHUSDT']
2026-07-14 16:48:58,748 INFO: 📊 Fetching missing performance data for 2 candidates...
2026-07-14 16:48:58,748 INFO: 🚫 Skipping BTCUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:58,748 INFO: 🚫 Skipping ETHUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:48:58,749 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:00,258 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:49:00,259 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): ETHUSDT
2026-07-14 16:49:00,259 WARNING: ⚠️ tickers was empty — restored 2 tickers from scanner
2026-07-14 16:49:00,259 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                                   [ 78%]
tests/test_supervisor.py::test_integration_real_stop_drops_bot 2026-07-14 16:49:00,268 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:00,278 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:00,278 INFO: 📋 Incubator ready: ['BTCUSDT']
2026-07-14 16:49:00,278 INFO: 📊 Fetching missing performance data for 2 candidates...
2026-07-14 16:49:00,279 INFO: ❌ Scored DROPPEDUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:49:00,279 INFO: ⚖️ Scored BTCUSDT: Net:50.00, Cyc:20, Score:7.6113
2026-07-14 16:49:00,279 INFO: 🧐 Evaluating REAL DROPPEDUSDT [Score: -inf, Profit: $0.00, Age: 24.00h] vs Candidate BTCUSDT [Score: 7.6113]
2026-07-14 16:49:00,280 INFO: ♻️ Substitution Triggered: Replacing DROPPEDUSDT with BTCUSDT (Score delta inf > Cushion 0.25)
2026-07-14 16:49:00,280 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:00,280 INFO: 🛑 Stopping Combat REAL process: DROPPEDUSDT (Rolling back to pure paper tracking)
2026-07-14 16:49:01,789 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:49:01,790 INFO: 🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): BTCUSDT
2026-07-14 16:49:01,790 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:49:01,790 INFO: Cycle Complete. REAL Swarm: ['BTCUSDT']
PASSED                                                                                                                                                                              [ 79%]
tests/test_supervisor.py::test_integration_authoritative_cleanup_stray 2026-07-14 16:49:01,801 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:01,803 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:01,804 INFO: 📋 Incubator ready: ['X']
2026-07-14 16:49:01,804 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:49:01,804 INFO: 🚫 Skipping X: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:49:01,804 WARNING: 🧹 Authoritative Cleanup: Found stray position for STRAYCOINUSDT (not in live_swarm or whitelist). Closing it!
2026-07-14 16:49:01,805 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:03,306 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): X
2026-07-14 16:49:03,307 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:49:03,308 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                      [ 79%]
tests/test_supervisor.py::test_integration_safety_trim 2026-07-14 16:49:03,316 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:03,318 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:03,319 INFO: 📋 Incubator ready: ['BOTA', 'BOTB']
2026-07-14 16:49:03,319 INFO: 📊 Fetching missing performance data for 2 candidates...
2026-07-14 16:49:03,319 INFO: ❌ Scored BOTA: Unprofitable. (Score: -INF)
2026-07-14 16:49:03,319 INFO: ❌ Scored BOTB: Unprofitable. (Score: -INF)
2026-07-14 16:49:03,319 INFO: ✂️ Reduced REAL swarm to 1 slots by removing least efficient bots.
2026-07-14 16:49:03,320 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:03,320 INFO: 🛑 Stopping Combat REAL process: BOTB (Rolling back to pure paper tracking)
2026-07-14 16:49:04,824 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BOTA
2026-07-14 16:49:04,825 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BOTB
2026-07-14 16:49:04,825 WARNING: ⚠️ tickers was empty — restored 2 tickers from scanner
2026-07-14 16:49:04,826 INFO: Cycle Complete. REAL Swarm: ['BOTA']
PASSED                                                                                                                                                                                      [ 79%]
tests/test_supervisor.py::test_integration_healed_tickers 2026-07-14 16:49:04,834 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:04,837 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:04,837 ERROR: ❌ Scanner returned nothing. Aborting cycle to prevent config wipe.
PASSED                                                                                                                                                                                   [ 80%]
tests/test_supervisor.py::test_integration_blacklisted_real_skipped 2026-07-14 16:49:04,846 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:04,855 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:04,855 INFO: 📋 Incubator ready: ['QUARANTINEDUSDT']
2026-07-14 16:49:04,856 INFO: 📊 Fetching missing performance data for 1 candidates...
2026-07-14 16:49:04,856 INFO: 🚫 Skipping QUARANTINEDUSDT: In REAL toxic blacklist. Quarantined.
2026-07-14 16:49:04,856 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:06,358 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): QUARANTINEDUSDT
2026-07-14 16:49:06,359 WARNING: ⚠️ tickers was empty — restored 1 tickers from scanner
2026-07-14 16:49:06,359 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                         [ 80%]
tests/test_supervisor_coverage.py::test_get_pm2_processes_empty_stdout PASSED                                                                                                                                                                      [ 80%]
tests/test_supervisor_coverage.py::test_get_pm2_processes_valid_json PASSED                                                                                                                                                                        [ 81%]
tests/test_supervisor_coverage.py::test_get_pm2_processes_exception 2026-07-14 16:49:06,368 ERROR: Критическая ошибка чтения стейта PM2: pm2 not found
PASSED                                                                                                                                                                         [ 81%]
tests/test_supervisor_coverage.py::test_get_pm2_processes_json_decode_error 2026-07-14 16:49:06,371 ERROR: Критическая ошибка чтения стейта PM2: Expecting value: line 1 column 1 (char 0)
PASSED                                                                                                                                                                 [ 81%]
tests/test_supervisor_coverage.py::test_reconcile_swarm_state_no_orphans 2026-07-14 16:49:06,374 WARNING: 🚨 [REAPER GUARD] Обнаружен рассинхрон. Orphaned процессы в режиме paper: {'paper-aliceusdt'}
2026-07-14 16:49:06,374 INFO: 💀 [HEAL] Процесс paper-aliceusdt успешно уничтожен.
PASSED                                                                                                                                                                    [ 81%]
tests/test_supervisor_coverage.py::test_reconcile_swarm_state_kills_orphans 2026-07-14 16:49:06,377 WARNING: 🚨 [REAPER GUARD] Обнаружен рассинхрон. Orphaned процессы в режиме paper: {'paper-aliceusdt', 'paper-orphonusdt'}
2026-07-14 16:49:06,377 INFO: 💀 [HEAL] Процесс paper-aliceusdt успешно уничтожен.
2026-07-14 16:49:06,378 INFO: 💀 [HEAL] Процесс paper-orphonusdt успешно уничтожен.
PASSED                                                                                                                                                                 [ 82%]
tests/test_supervisor_coverage.py::test_reconcile_swarm_state_empty_pm2 PASSED                                                                                                                                                                     [ 82%]
tests/test_supervisor_coverage.py::test_stop_bot_filenotfound 2026-07-14 16:49:06,383 ERROR: PM2 binary not found — cannot delete real-btc
PASSED                                                                                                                                                                               [ 82%]
tests/test_supervisor_coverage.py::test_stop_bot_oserror 2026-07-14 16:49:06,386 ERROR: OS error deleting paper-eth: permission denied
PASSED                                                                                                                                                                                    [ 83%]
tests/test_supervisor_coverage.py::test_stop_bot_general_exception 2026-07-14 16:49:06,389 ERROR: Failed to delete real-sol: unexpected
PASSED                                                                                                                                                                          [ 83%]
tests/test_supervisor_coverage.py::test_enforce_invariant_gate_adds_exposed_ticker 2026-07-14 16:49:06,391 WARNING: 🛡️ Invariant Protection Gate: Forced retention of NEWCOINUSDT in live_swarm due to active exposure ($100.00 USDT).
PASSED                                                                                                                                                          [ 83%]
tests/test_supervisor_coverage.py::test_enforce_invariant_gate_dust_filtered PASSED                                                                                                                                                                [ 84%]
tests/test_supervisor_coverage.py::test_enforce_invariant_gate_empty_positions PASSED                                                                                                                                                              [ 84%]
tests/test_supervisor_coverage.py::test_enforce_invariant_gate_zero_qty_filtered PASSED                                                                                                                                                            [ 84%]
tests/test_supervisor_coverage.py::test_enforce_invariant_gate_exception 2026-07-14 16:49:06,400 ERROR: Failed to verify exchange exposure tracking in Invariant Gate: exchange down
PASSED                                                                                                                                                                    [ 85%]
tests/test_supervisor_coverage.py::test_reset_bot_state_files_rebases_on_trailing_stop 2026-07-14 16:49:06,407 WARNING: 📉 Обнаружен Trailing Stop для BTCUSDT. Капитал ребазирован: 115.0 -> 200.0
2026-07-14 16:49:06,408 INFO: 💾 Archived paper state for BTCUSDT: paper_state_BTCUSDT.json -> archive_BTCUSDT_1784040546_paper_state_BTCUSDT.json
2026-07-14 16:49:06,409 INFO: 🗑️ Deleted stale state file: paper_state_BTCUSDT.json
2026-07-14 16:49:06,412 INFO: ✨ Reset paper state files for BTCUSDT to clean initial values (capital: 200.0).
PASSED                                                                                                                                                      [ 85%]
tests/test_supervisor_coverage.py::test_reset_bot_state_files_no_state_uses_config 2026-07-14 16:49:06,428 INFO: ✨ Reset paper state files for ETHUSDT to clean initial values (capital: 150.0).
PASSED                                                                                                                                                          [ 85%]
tests/test_supervisor_coverage.py::test_start_bot_real_no_state_file PASSED                                                                                                                                                                        [ 85%]
tests/test_supervisor_coverage.py::test_start_bot_real_with_existing_state PASSED                                                                                                                                                                  [ 86%]
tests/test_supervisor_coverage.py::test_start_bot_paper_always_resets PASSED                                                                                                                                                                       [ 86%]
tests/test_supervisor_coverage.py::test_ensure_real_bots_alive_missing_bot_restarts 2026-07-14 16:49:06,457 WARNING: 🔄 Real bot BTCUSDT is in live_swarm but not running in PM2. Restarting...
2026-07-14 16:49:06,457 INFO: ✅ Restarted real bot BTCUSDT
PASSED                                                                                                                                                         [ 86%]
tests/test_supervisor_coverage.py::test_ensure_real_bots_alive_all_running PASSED                                                                                                                                                                  [ 87%]
tests/test_supervisor_coverage.py::test_ensure_real_bots_alive_empty_swarm PASSED                                                                                                                                                                  [ 87%]
tests/test_supervisor_coverage.py::test_ensure_real_bots_alive_start_exception 2026-07-14 16:49:06,463 WARNING: 🔄 Real bot BTCUSDT is in live_swarm but not running in PM2. Restarting...
2026-07-14 16:49:06,464 ERROR: ❌ Failed to restart real bot BTCUSDT: pm2 crash
PASSED                                                                                                                                                              [ 87%]
tests/test_supervisor_coverage.py::test_enforce_swarm_consistency_no_positions 2026-07-14 16:49:06,466 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
PASSED                                                                                                                                                              [ 88%]
tests/test_supervisor_coverage.py::test_enforce_swarm_consistency_heal_failure 2026-07-14 16:49:06,472 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
2026-07-14 16:49:06,472 WARNING: ⚠️ Unauthorized stray position for ZECUSDT (allowed pools are empty). Scheduled for liquidation.
2026-07-14 16:49:06,473 INFO: 🧹 Closing all positions for unauthorized ticker: ZECUSDT
2026-07-14 16:49:06,553 INFO: ✅ Positions for ZECUSDT closed successfully.
PASSED                                                                                                                                                              [ 88%]
tests/test_supervisor_new.py::test_calculate_bot_score 2026-07-14 16:49:06,555 INFO: 🛡️ Scored BTCUSDT: REAL bot in drawdown. LOCKED IN COMBAT (Score: INF).
2026-07-14 16:49:06,555 INFO: ❌ Scored BTCUSDT: Rejected (Cycles 5 < 10). (Score: -INF)
2026-07-14 16:49:06,555 INFO: ❌ Scored BTCUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:49:06,555 INFO: ❌ Scored BTCUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:49:06,556 INFO: ⚖️ Scored BTCUSDT: Net:100.00, Cyc:20, Score:15.2226
2026-07-14 16:49:06,556 INFO: ⚖️ Scored BTCUSDT: Net:100.00, Cyc:20, Score:18.2671
PASSED                                                                                                                                                                                      [ 88%]
tests/test_supervisor_new.py::test_calc_rotation_score PASSED                                                                                                                                                                                      [ 88%]
tests/test_supervisor_new.py::test_selective_merge_incubator 2026-07-14 16:49:06,558 INFO: 🚀 First-start: launching 4 bots from scanner: ['A', 'B', 'C', 'D']
2026-07-14 16:49:06,559 INFO: 🔄 Rotation: keep=2, add=2, remove=2
2026-07-14 16:49:06,559 INFO: 🔄   removing (worst by score): ['C', 'D']
2026-07-14 16:49:06,559 INFO: 🔄   adding (best from scanner): ['E', 'F']
2026-07-14 16:49:06,559 INFO: 📋 Final incubator (4 bots): ['A', 'B', 'E', 'F']
PASSED                                                                                                                                                                                [ 89%]
tests/test_supervisor_new.py::test_get_bot_efficiency PASSED                                                                                                                                                                                       [ 89%]
tests/test_supervisor_new.py::test_reset_bot_state_files_real 2026-07-14 16:49:06,564 INFO: 💾 Archived state for BTCUSDT: real_state_BTCUSDT.json -> archive_BTCUSDT_1784040546_real_state_BTCUSDT.json
2026-07-14 16:49:06,564 INFO: 🗑️ Deleted stale state file: real_state_BTCUSDT.json
2026-07-14 16:49:06,565 INFO: 💾 Archived state for BTCUSDT: shadow_state_BTCUSDT.json -> archive_BTCUSDT_1784040546_shadow_state_BTCUSDT.json
2026-07-14 16:49:06,565 INFO: 🗑️ Deleted stale state file: shadow_state_BTCUSDT.json
2026-07-14 16:49:06,566 INFO: ✨ Reset real state files for BTCUSDT to clean initial values (capital: 200.0).
PASSED                                                                                                                                                                               [ 89%]
tests/test_supervisor_new.py::test_reset_bot_state_files_paper 2026-07-14 16:49:06,570 INFO: 💾 Archived paper state for BTCUSDT: paper_state_BTCUSDT.json -> archive_BTCUSDT_1784040546_paper_state_BTCUSDT.json
2026-07-14 16:49:06,570 INFO: 🗑️ Deleted stale state file: paper_state_BTCUSDT.json
2026-07-14 16:49:06,570 INFO: 💾 Archived paper state for BTCUSDT: paper_shadow_BTCUSDT.json -> archive_BTCUSDT_1784040546_paper_shadow_BTCUSDT.json
2026-07-14 16:49:06,571 INFO: 🗑️ Deleted stale state file: paper_shadow_BTCUSDT.json
2026-07-14 16:49:06,571 INFO: ✨ Reset paper state files for BTCUSDT to clean initial values (capital: 150.0).
PASSED                                                                                                                                                                              [ 90%]
tests/test_supervisor_new.py::test_enforce_swarm_consistency 2026-07-14 16:49:06,574 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
2026-07-14 16:49:06,574 WARNING: ⚠️ Unauthorized position found for ETHUSDT (not in live_swarm or real_whitelist). Scheduled for liquidation.
2026-07-14 16:49:06,574 INFO: 🧹 Closing all positions for unauthorized ticker: ETHUSDT
2026-07-14 16:49:06,575 INFO: ✅ Positions for ETHUSDT closed successfully.
PASSED                                                                                                                                                                                [ 90%]
tests/test_supervisor_new.py::test_ensure_real_bots_alive 2026-07-14 16:49:06,577 WARNING: 🔄 Real bot ETHUSDT is in live_swarm but not running in PM2. Restarting...
2026-07-14 16:49:06,578 INFO: ✅ Restarted real bot ETHUSDT
PASSED                                                                                                                                                                                   [ 90%]
tests/test_supervisor_new.py::test_manage_swarm_toxic_flow 2026-07-14 16:49:06,585 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:06,586 CRITICAL: 🚫 STOP signal: TRXUSDT (paper) → toxic_blacklist_paper + black_list (expires Tue Jul 14 17:17:54 2026)
2026-07-14 16:49:06,587 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:06,588 INFO: ⏳ TRXUSDT is in Toxic Quarantine. Skipping.
2026-07-14 16:49:06,588 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['BTCUSDT']
2026-07-14 16:49:06,588 INFO: 🔄 Rotation: keep=1, add=0, remove=0
2026-07-14 16:49:06,588 INFO: 📋 Final incubator (1 bots): ['BTCUSDT']
2026-07-14 16:49:06,588 INFO: 📋 Incubator ready: ['BTCUSDT']
2026-07-14 16:49:06,588 INFO: 🚫 Skipping BTCUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:49:06,589 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:06,589 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:49:06,589 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:49:08,095 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:49:08,096 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                                  [ 91%]
tests/test_supervisor_new.py::test_isolated_blacklists_signal_processing 2026-07-14 16:49:08,102 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:08,102 CRITICAL: 🚫 STOP signal: BTCUSDT (paper) → toxic_blacklist_paper + black_list (expires Tue Jul 14 17:17:56 2026)
2026-07-14 16:49:08,103 INFO: ✅ EXIT signal: ETHUSDT (real) → probation_real (expires Tue Jul 14 17:48:10 2026)
2026-07-14 16:49:08,104 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:08,104 INFO: ⏳ BTCUSDT is in Toxic Quarantine. Skipping.
2026-07-14 16:49:08,104 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['ETHUSDT']
2026-07-14 16:49:08,104 INFO: 🔄 Rotation: keep=1, add=0, remove=0
2026-07-14 16:49:08,104 INFO: 📋 Final incubator (1 bots): ['ETHUSDT']
2026-07-14 16:49:08,104 INFO: 📋 Incubator ready: ['ETHUSDT']
2026-07-14 16:49:08,105 INFO: 🚫 Skipping ETHUSDT: Not in real_whitelist and not a currently running real bot.
2026-07-14 16:49:08,105 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:10,094 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): ETHUSDT
2026-07-14 16:49:10,343 INFO: Cycle Complete. REAL Swarm: []
PASSED                                                                                                                                                                    [ 91%]
tests/test_supervisor_new.py::test_get_running_bots_info PASSED                                                                                                                                                                                    [ 91%]
tests/test_supervisor_new.py::test_stop_bot PASSED                                                                                                                                                                                                 [ 92%]
tests/test_supervisor_new.py::test_start_bot PASSED                                                                                                                                                                                                [ 92%]
tests/test_supervisor_new.py::test_rotation_guard_immature_unprofitable_protected 2026-07-14 16:49:10,353 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['B']
2026-07-14 16:49:10,353 INFO: 🔄 Rotation: keep=2, add=2, remove=2
2026-07-14 16:49:10,354 INFO: 🔄   removing (worst by score): ['C', 'D']
2026-07-14 16:49:10,354 INFO: 🔄   adding (best from scanner): ['E', 'F']
2026-07-14 16:49:10,354 INFO: 📋 Final incubator (4 bots): ['A', 'B', 'E', 'F']
PASSED                                                                                                                                                           [ 92%]
tests/test_supervisor_new.py::test_rotation_guard_immature_profitable_protected 2026-07-14 16:49:10,356 INFO: 🔄 Rotation: keep=1, add=2, remove=2
2026-07-14 16:49:10,356 INFO: 🔄   removing (worst by score): ['B', 'C']
2026-07-14 16:49:10,357 INFO: 🔄   adding (best from scanner): ['D', 'E']
2026-07-14 16:49:10,357 INFO: 📋 Final incubator (3 bots): ['A', 'D', 'E']
PASSED                                                                                                                                                             [ 92%]
tests/test_supervisor_new.py::test_rotation_guard_mature_unprofitable_replaced 2026-07-14 16:49:10,359 INFO: 🔄 Rotation: keep=1, add=2, remove=2
2026-07-14 16:49:10,359 INFO: 🔄   removing (worst by score): ['B', 'C']
2026-07-14 16:49:10,359 INFO: 🔄   adding (best from scanner): ['D', 'E']
2026-07-14 16:49:10,360 INFO: 📋 Final incubator (3 bots): ['A', 'D', 'E']
PASSED                                                                                                                                                              [ 93%]
tests/test_supervisor_new.py::test_rotation_guard_default_min_cycles_for_rotation 2026-07-14 16:49:10,362 INFO: 🛡️ Rotation Guard: Protecting immature bots (cycles < 6): ['A']
2026-07-14 16:49:10,362 INFO: 🔄 Rotation: keep=1, add=2, remove=2
2026-07-14 16:49:10,362 INFO: 🔄   removing (worst by score): ['B', 'C']
2026-07-14 16:49:10,362 INFO: 🔄   adding (best from scanner): ['D', 'E']
2026-07-14 16:49:10,363 INFO: 📋 Final incubator (3 bots): ['A', 'D', 'E']
PASSED                                                                                                                                                           [ 93%]
tests/test_supervisor_new.py::test_reset_bot_state_files_exceptions 2026-07-14 16:49:10,366 ERROR: Failed to archive C:\Python\Prosperous_Bot\futures_portfolio\real_state_BTCUSDT.json: Copy failed
2026-07-14 16:49:10,366 ERROR: Failed to archive C:\Python\Prosperous_Bot\futures_portfolio\shadow_state_BTCUSDT.json: Copy failed
2026-07-14 16:49:10,366 INFO: ✨ Reset real state files for BTCUSDT to clean initial values (capital: 200.0).
PASSED                                                                                                                                                                         [ 93%]
tests/test_supervisor_new.py::test_manage_swarm_rotation_hysteresis_and_cushion 2026-07-14 16:49:10,370 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:10,371 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:10,371 INFO: 🔄 Rotation: keep=1, add=0, remove=1
2026-07-14 16:49:10,371 INFO: 🔄   removing (worst by score): ['ETHUSDT']
2026-07-14 16:49:10,371 INFO: 📋 Final incubator (1 bots): ['BTCUSDT']
2026-07-14 16:49:10,371 INFO: 📋 Incubator ready: ['BTCUSDT']
2026-07-14 16:49:10,372 INFO: ⚖️ Scored BTCUSDT: Net:100.00, Cyc:10, Score:23.9790
2026-07-14 16:49:10,372 INFO: ❌ Scored ETHUSDT: Unprofitable. (Score: -INF)
2026-07-14 16:49:10,372 INFO: 🧐 Evaluating REAL ETHUSDT [Score: -inf, Profit: $0.00, Age: 0.00h] vs Candidate BTCUSDT [Score: 23.9790]
2026-07-14 16:49:10,372 INFO: ♻️ Substitution Triggered: Replacing ETHUSDT with BTCUSDT (Score delta inf > Cushion 0.2)
2026-07-14 16:49:10,372 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:10,856 INFO: 🛑 Stopping Combat REAL process: ETHUSDT (Rolling back to pure paper tracking)
2026-07-14 16:49:16,370 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:49:16,371 INFO: 🔥 [A] LAUNCHING PARALLEL COMBAT (REAL): BTCUSDT
2026-07-14 16:49:16,616 INFO: Cycle Complete. REAL Swarm: ['BTCUSDT']
2026-07-14 16:49:16,616 WARNING: 🔄 Real bot BTCUSDT is in live_swarm but not running in PM2. Restarting...
2026-07-14 16:49:16,616 INFO: ✅ Restarted real bot BTCUSDT
PASSED                                                                                                                                                             [ 94%]
tests/test_supervisor_rotation_whitelist.py::test_manage_swarm_respects_whitelist_during_rotation 2026-07-14 16:49:16,620 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:16,620 INFO: 🗑️ BTCUSDT removed from live_swarm (signal: exit)
2026-07-14 16:49:16,620 INFO: ✅ EXIT signal: BTCUSDT (paper) → probation_paper (expires Tue Jul 14 17:48:19 2026)
2026-07-14 16:49:16,621 CRITICAL: 🚫 STOP signal: BTCUSDT (paper) → toxic_blacklist_paper + black_list (expires Tue Jul 14 17:18:04 2026)
2026-07-14 16:49:16,622 CRITICAL: 🚫 STOP signal: BTCUSDT (real) → toxic_blacklist_real + black_list (expires Tue Jul 14 17:18:04 2026)
2026-07-14 16:49:16,623 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:16,623 INFO: 🔄 Rotation: keep=2, add=0, remove=0
2026-07-14 16:49:16,624 INFO: 📋 Final incubator (2 bots): ['BTCUSDT', 'ETHUSDT']
2026-07-14 16:49:16,624 INFO: 📋 Incubator ready: ['BTCUSDT', 'ETHUSDT']
2026-07-14 16:49:16,624 INFO: ⚖️ Scored BTCUSDT: Net:10.00, Cyc:20, Score:1.8267
2026-07-14 16:49:16,624 INFO: ⚖️ Scored ETHUSDT: Net:10.00, Cyc:20, Score:1.8267
2026-07-14 16:49:16,624 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:16,625 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:49:16,625 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:49:18,134 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:49:18,135 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): ETHUSDT
2026-07-14 16:49:18,135 INFO: Cycle Complete. REAL Swarm: ['BTCUSDT', 'ETHUSDT']
2026-07-14 16:49:18,136 INFO: --- Starting Pure Live Supervisor (Strict Scanner Sync) ---
2026-07-14 16:49:18,136 INFO: 🔍 Running ticker scanner (Min Vol: 20M)...
2026-07-14 16:49:18,137 INFO: 🔄 Rotation: keep=2, add=0, remove=0
2026-07-14 16:49:18,137 INFO: 📋 Final incubator (2 bots): ['BTCUSDT', 'ETHUSDT']
2026-07-14 16:49:18,137 INFO: 📋 Incubator ready: ['BTCUSDT', 'ETHUSDT']
2026-07-14 16:49:18,137 INFO: ⚖️ Scored BTCUSDT: Net:10.00, Cyc:20, Score:1.8267
2026-07-14 16:49:18,137 INFO: ⚖️ Scored ETHUSDT: Net:10.00, Cyc:20, Score:1.8267
2026-07-14 16:49:18,137 INFO: ⚔️ [REAPER GUARD] Phase: Reconciling PM2 state with target configuration.
2026-07-14 16:49:18,138 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:49:18,138 ERROR: Критическая ошибка чтения стейта PM2: not enough values to unpack (expected 2, got 0)
2026-07-14 16:49:19,645 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): BTCUSDT
2026-07-14 16:49:19,646 INFO: 🚀 [A] Launching Continuous Incubator (PAPER): ETHUSDT
2026-07-14 16:49:19,646 INFO: Cycle Complete. REAL Swarm: ['BTCUSDT', 'ETHUSDT']
PASSED                                                                                                                                           [ 94%]
tests/test_supervisor_service.py::test_get_sleep_interval_valid PASSED                                                                                                                                                                             [ 94%]
tests/test_supervisor_service.py::test_get_sleep_interval_missing_file PASSED                                                                                                                                                                      [ 95%]
tests/test_supervisor_service.py::test_get_sleep_interval_floor PASSED                                                                                                                                                                             [ 95%]
tests/test_supervisor_service.py::test_supervisor_service_main PASSED                                                                                                                                                                              [ 95%]
tests/test_supervisor_whitelist.py::test_enforce_swarm_consistency_respects_whitelist 2026-07-14 16:49:19,661 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
2026-07-14 16:49:19,661 WARNING: ⚠️ Unauthorized position found for SOLUSDT (not in live_swarm or real_whitelist). Scheduled for liquidation.
2026-07-14 16:49:19,661 INFO: 🧹 Closing all positions for unauthorized ticker: SOLUSDT
2026-07-14 16:49:19,662 INFO: ✅ Positions for SOLUSDT closed successfully.
PASSED                                                                                                                                                       [ 96%]
tests/test_supervisor_whitelist.py::test_enforce_swarm_consistency_all_allowed 2026-07-14 16:49:19,664 INFO: 🛡️ Enforcing Swarm Consistency: Checking for unauthorized positions.
PASSED                                                                                                                                                              [ 96%]
tests/test_swarm_analyzer.py::test_analyze_swarm PASSED                                                                                                                                                                                            [ 96%]
tests/test_swarm_analyzer.py::test_analyze_swarm_missing_config PASSED                                                                                                                                                                             [ 96%]
tests/test_telegram_sender.py::test_send_to_telegram_success PASSED                                                                                                                                                                                [ 97%]
tests/test_telegram_sender.py::test_send_to_telegram_rate_limit PASSED                                                                                                                                                                             [ 97%]
tests/test_telegram_sender.py::test_send_to_telegram_rate_limit_invalid_json PASSED                                                                                                                                                                [ 97%]
tests/test_telegram_sender.py::test_send_to_telegram_timeout PASSED                                                                                                                                                                                [ 98%]
tests/test_telegram_sender.py::test_send_to_telegram_connection_error PASSED                                                                                                                                                                       [ 98%]
tests/test_telegram_sender.py::test_worker_missing_credentials PASSED                                                                                                                                                                              [ 98%]
tests/test_telegram_sender.py::test_worker_empty_queue PASSED                                                                                                                                                                                      [ 99%]
tests/test_telegram_sender.py::test_worker_send_failure PASSED                                                                                                                                                                                     [ 99%]
tests/test_telegram_sender.py::test_send_to_telegram_api_error PASSED                                                                                                                                                                              [ 99%]
tests/test_telegram_sender.py::test_worker_success PASSED                                                                                                                                                                                          [100%]

================================================================================================================= 327 passed in 52.59s ==================================================================================================================
(.venv) (base) PS C:\Python\Prosperous_Bot\futures_portfolio> 