import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import glob
from datetime import datetime

def parse_rebalance_log(file_path):
    data = []
    rebalances = []

    # Regex patterns
    # New detailed heartbeat: TPV=114.86 | PnL=-0.14 | ARUSDT=2.263 | L:26.8% [-0.2%] {+30.77$} | S:34.3% [+0.3%] {+39.40$} | V:35.0% [+0.0%] {+40.21$} | C:3.9% {4.50$}
    heartbeat_full_ptrn = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: Heartbeat: "
        r"TPV=([\d.]+) \| PnL=([+\d.-]+) \| ([A-Z]+)=([\d.]+) \| "
        r"L:([\d.]+)%.*?\| S:([\d.]+)%.*?\| V:([\d.]+)%.*?\| C:([\d.]+)%"
    )
    # Simplified heartbeat (no weights): TPV=114.86 | PnL=-0.14 | ARUSDT=2.25576 | Cycles=1
    heartbeat_simple_ptrn = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: Heartbeat: "
        r"TPV=([\d.]+) \| PnL=([+\d.-]+) \| ([A-Z]+)=([\d.]+) \| Cycles=(\d+)"
    )
    # Old-style heartbeat (Balance=...): Balance=xxx | SAFE:xxx | TICKER=price | L:x% S:y% V:z% C:w%
    heartbeat_old_ptrn = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: Heartbeat: "
        r"Balance=([\d.]+) \| (?:SAFE:([\d.]+) \| )?([A-Z]+)=([\d.e-]+) \| "
        r"L:([\d.]+)% S:([\d.]+)% V:([\d.]+)% C:([\d.]+)%"
    )
    tpv_update_ptrn = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: Rebalance #(\d+) complete\. TPV: ([\d.]+)"
    )
    # TPV standalone line (no timestamp — use last known timestamp)
    tpv_standalone_ptrn = re.compile(r"^TPV: ([\d.]+)")
    # Trade: 📝 PAPER: BUY 68.6 ARUSDT_LONG @ 2.263  OR  PAPER: SELL ...
    trade_ptrn = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: .*PAPER: (BUY|SELL) ([\d.]+) ([A-Z_]+) @ ([\d.eE+-]+)"
    )
    safe_activated_ptrn = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO .* SAFE ACTIVATED: Siphoned ([\d.]+) USDT\. New Reserve: ([\d.]+)"
    )

    if not os.path.exists(file_path):
        print(f"Log file not found: {file_path}")
        return None, None

    last_ts = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 0. Standalone TPV line (no timestamp)
            m = tpv_standalone_ptrn.match(line)
            if m and last_ts:
                tpv_val = float(m.group(1))
                data.append({
                    'timestamp': last_ts,
                    'tpv': tpv_val,
                    'type': 'tpv_standalone'
                })
                continue

            # 1. Full heartbeat with weights
            m = heartbeat_full_ptrn.search(line)
            if m:
                ts_str, tpv, pnl, ticker, price, l_w, s_w, v_w, c_w = m.groups()
                ts = pd.to_datetime(ts_str)
                last_ts = ts
                data.append({
                    'timestamp': ts,
                    'tpv': float(tpv),
                    'pnl': float(pnl),
                    'price': float(price),
                    'L': float(l_w),
                    'S': float(s_w),
                    'V': float(v_w),
                    'C': float(c_w),
                    'type': 'heartbeat_full'
                })
                continue

            # 2. Simple heartbeat (no weights)
            m = heartbeat_simple_ptrn.search(line)
            if m:
                ts_str, tpv, pnl, ticker, price, cycles = m.groups()
                ts = pd.to_datetime(ts_str)
                last_ts = ts
                data.append({
                    'timestamp': ts,
                    'tpv': float(tpv),
                    'pnl': float(pnl),
                    'price': float(price),
                    'cycle': int(cycles),
                    'type': 'heartbeat_simple'
                })
                continue

            # 3. Old-style heartbeat
            m = heartbeat_old_ptrn.search(line)
            if m:
                ts_str, balance, safe_bal, ticker, price, l_w, s_w, v_w, c_w = m.groups()
                ts = pd.to_datetime(ts_str)
                last_ts = ts
                data.append({
                    'timestamp': ts,
                    'balance': float(balance),
                    'safe': float(safe_bal) if safe_bal else 0.0,
                    'price': float(price),
                    'L': float(l_w),
                    'S': float(s_w),
                    'V': float(v_w),
                    'C': float(c_w),
                    'type': 'heartbeat'
                })
                continue

            # 4. SAFE Activated Event
            m = safe_activated_ptrn.search(line)
            if m:
                ts_str, siphoned, reserve = m.groups()
                ts = pd.to_datetime(ts_str)
                last_ts = ts
                rebalances.append({
                    'timestamp': ts,
                    'action': 'SAFE_SIPHON',
                    'qty': float(siphoned),
                    'asset': 'SAFE',
                    'price': float(reserve),
                    'type': 'safe_event'
                })
                continue

            # 5. TPV update after rebalance
            m = tpv_update_ptrn.search(line)
            if m:
                ts_str, cycle, tpv = m.groups()
                ts = pd.to_datetime(ts_str)
                last_ts = ts
                data.append({
                    'timestamp': ts,
                    'tpv': float(tpv),
                    'cycle': int(cycle),
                    'type': 'tpv_update'
                })
                continue

            # 6. Trades
            m = trade_ptrn.search(line)
            if m:
                ts_str, action, qty, asset, price = m.groups()
                ts = pd.to_datetime(ts_str)
                last_ts = ts
                rebalances.append({
                    'timestamp': ts,
                    'action': action,
                    'qty': float(qty),
                    'asset': asset,
                    'price': float(price)
                })
                continue

    if not data:
        return None, None

    # Ensure all records have the same keys (fill missing with None)
    all_keys = set()
    for d in data:
        all_keys.update(d.keys())
    for d in data:
        for k in all_keys:
            d.setdefault(k, None)

    df = pd.DataFrame(data)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')

    # Handle outliers in TPV to prevent scale distortion
    if not df.empty:
        initial_tpv = df[df['tpv'].notna()]['tpv'].iloc[0] if not df[df['tpv'].notna()].empty else 65.0
        df.loc[(df['tpv'] < initial_tpv * 0.5) | (df['tpv'] > initial_tpv * 2.0), 'tpv'] = None

    # Forward fill TPV and Price to have them on all rows
    df['tpv'] = df['tpv'].ffill()
    if 'price' in df.columns:
        df['price'] = df['price'].ffill()
    if 'balance' in df.columns:
        df['balance'] = df['balance'].ffill()

    return df, pd.DataFrame(rebalances)

def visualize_bot(log_path):
    base = os.path.basename(log_path).replace(".log", "")
    # Support both "rebalance_TICKER" and "paper_TICKER" naming
    ticker = base.replace("rebalance_", "").replace("paper_", "")
    df, df_trades = parse_rebalance_log(log_path)
    
    if df is None or df.empty:
        print(f"No data found in {log_path}")
        return

    # Create figure with subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3],
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

    # 1. TPV (Main Plot)
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['tpv'], name="TPV", line=dict(color='royalblue', width=2)),
        row=1, col=1, secondary_y=False
    )
    
    # 2. Price (Main Plot, Secondary Y)
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['price'], name=f"{ticker} Price", 
                   line=dict(color='orange', width=1, dash='dot'), opacity=0.5),
        row=1, col=1, secondary_y=True
    )

    # 3. Trades as markers
    if df_trades is not None and not df_trades.empty:
        # 3.1 SAFE Activation Markers (Overlayed on TPV line)
        safe_mask = df_trades['action'] == 'SAFE_SIPHON'
        if safe_mask.any():
            # Merge with df to get the TPV at the time of SAFE activation
            safe_events = pd.merge_asof(
                df_trades[safe_mask].sort_values('timestamp'),
                df[['timestamp', 'tpv']].sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
            
            fig.add_trace(
                go.Scatter(x=safe_events['timestamp'], y=safe_events['tpv'],
                           mode='markers+text', name="SAFE Activated",
                           marker=dict(color='gold', symbol='star', size=15, line=dict(width=1, color='white')),
                           text=safe_events['qty'].apply(lambda x: f"+{x:.2f}"),
                           textposition="top center",
                           hovertext=safe_events.apply(lambda x: f"Siphoned: {x['qty']} USDT | Total Reserve: {x['price']}", axis=1)),
                row=1, col=1, secondary_y=False
            )

        # 3.2 Regular Trades
        asset_types = {
            'LONG': {'color': 'green', 'label': 'Long Asset'},
            'SHORT': {'color': 'red', 'label': 'Short Asset'}
        }
        
        for asset_suffix, config in asset_types.items():
            mask = df_trades['asset'].str.contains(asset_suffix)
            if not mask.any(): continue
            
            # Separate BUY and SELL within this asset type for correct arrow direction
            for action in ['BUY', 'SELL']:
                action_mask = mask & (df_trades['action'] == action)
                if not action_mask.any(): continue
                
                symbol = 'triangle-up' if action == 'BUY' else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(x=df_trades[action_mask]['timestamp'], y=df_trades[action_mask]['price'],
                               mode='markers', 
                               name=f"{config['label']} {action}",
                               marker=dict(color=config['color'], symbol=symbol, size=10, 
                                          line=dict(width=1, color='white')),
                               hovertext=df_trades[action_mask].apply(
                                   lambda x: f"{x['action']} {x['qty']} {x['asset']} @ {x['price']}", axis=1)),
                    row=1, col=1, secondary_y=True
                )

    # 4. Weights (Bottom Plot)
    for col, color in zip(['L', 'S', 'V', 'C'], ['blue', 'red', 'purple', 'gray']):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[col], name=f"Weight {col}%", stackgroup='one', line=dict(color=color)),
                row=2, col=1
            )

    # Update layout and fix Y-axis scales
    tpv_min = df['tpv'].min()
    tpv_max = df['tpv'].max()
    tpv_range = tpv_max - tpv_min
    
    price_min = df['price'].min()
    price_max = df['price'].max()
    price_range = price_max - price_min

    fig.update_layout(
        title=f"Detailed Analysis: {ticker} (Source: {os.path.basename(log_path)})",
        xaxis_title="Time",
        yaxis=dict(
            title="TPV (USDT)",
            range=[tpv_min - (tpv_range * 0.1), tpv_max + (tpv_range * 0.2)] # Extra space at top for text
        ),
        yaxis2=dict(
            title="Asset Price",
            range=[price_min - (price_range * 0.05), price_max + (price_range * 0.05)],
            overlaying='y',
            side='right'
        ),
        yaxis3_title="Weights %",
        height=800,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    output_path = os.path.join(os.path.dirname(log_path), f"visualize_{ticker}.html")
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Auto-detect first paper_ log
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        candidates = sorted(glob.glob(os.path.join(log_dir, "paper_*.log")))
        if candidates:
            log_file = candidates[0]
            print(f"Auto-selected: {log_file}")
        else:
            log_file = os.path.join(log_dir, "rebalance_DOGSUSDT.log")
    
    visualize_bot(log_file)
