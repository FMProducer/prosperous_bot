import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

def parse_rebalance_log(file_path):
    data = []
    rebalances = []
    
    # Regex patterns
    heartbeat_ptrn = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO Heartbeat: Balance=([\d.]+) \| (?:SAFE:([\d.]+) \| )?([A-Z]+)=([\d.e-]+) \| L:([\d.]+)% S:([\d.]+)% V:([\d.]+)% C:([\d.]+)%")
    tpv_update_ptrn = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO Rebalance #(\d+) complete\. TPV: ([\d.]+)")
    trade_ptrn = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO .* PAPER: (BUY|SELL) ([\d.]+) ([A-Z_]+) @ ([\d.e-]+)")
    tpv_heartbeat_ptrn = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO Heartbeat: TPV=([\d.]+) \| PnL=([+\d.-]+) \| Cycles=(\d+)")
    safe_activated_ptrn = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO .* SAFE ACTIVATED: Siphoned ([\d.]+) USDT\. New Reserve: ([\d.]+)")

    if not os.path.exists(file_path):
        print(f"Log file not found: {file_path}")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Heartbeat with weights and SAFE
            m = heartbeat_ptrn.search(line)
            if m:
                ts, balance, safe_bal, ticker, price, l_weight, s_weight, v_weight, c_weight = m.groups()
                data.append({
                    'timestamp': pd.to_datetime(ts),
                    'balance': float(balance),
                    'safe': float(safe_bal) if safe_bal else 0.0,
                    'price': float(price),
                    'L': float(l_weight),
                    'S': float(s_weight),
                    'V': float(v_weight),
                    'C': float(c_weight),
                    'type': 'heartbeat'
                })
                continue
            
            # 5. SAFE Activated Event
            m = safe_activated_ptrn.search(line)
            if m:
                ts, siphoned, reserve = m.groups()
                rebalances.append({
                    'timestamp': pd.to_datetime(ts),
                    'action': 'SAFE_SIPHON',
                    'qty': float(siphoned),
                    'asset': 'SAFE',
                    'price': float(reserve), # Store reserve as "price" for TPV plot positioning
                    'type': 'safe_event'
                })
                continue
            
            # 2. TPV update after rebalance
            m = tpv_update_ptrn.search(line)
            if m:
                ts, cycle, tpv = m.groups()
                data.append({
                    'timestamp': pd.to_datetime(ts),
                    'tpv': float(tpv),
                    'cycle': int(cycle),
                    'type': 'tpv_update'
                })
                continue

            # 3. TPV heartbeat
            m = tpv_heartbeat_ptrn.search(line)
            if m:
                ts, tpv, pnl, cycles = m.groups()
                data.append({
                    'timestamp': pd.to_datetime(ts),
                    'tpv': float(tpv),
                    'pnl': float(pnl),
                    'cycle': int(cycles),
                    'type': 'tpv_heartbeat'
                })
                continue
            
            # 4. Trades
            m = trade_ptrn.search(line)
            if m:
                ts, action, qty, asset, price = m.groups()
                rebalances.append({
                    'timestamp': pd.to_datetime(ts),
                    'action': action,
                    'qty': float(qty),
                    'asset': asset,
                    'price': float(price)
                })

    if not data:
        return None, None

    df = pd.DataFrame(data).sort_values('timestamp')
    
    # 1. Handle outliers in TPV to prevent scale distortion
    if not df.empty:
        # Initial TPV is usually around the first valid TPV
        initial_tpv = df[df['tpv'].notna()]['tpv'].iloc[0] if not df[df['tpv'].notna()].empty else 65.0
        
        # Define reasonable bounds (e.g., 50% to 200% of initial)
        # Values outside this are likely parsing errors or transient state resets
        df.loc[(df['tpv'] < initial_tpv * 0.5) | (df['tpv'] > initial_tpv * 2.0), 'tpv'] = None

    # Forward fill TPV and Price to have them on all rows
    df['tpv'] = df['tpv'].ffill()
    df['price'] = df['price'].ffill()
    df['balance'] = df['balance'].ffill()
    
    return df, pd.DataFrame(rebalances)

def visualize_bot(log_path):
    ticker = os.path.basename(log_path).replace("rebalance_", "").replace(".log", "")
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
        # Default for testing
        log_file = "logs/rebalance_DOGSUSDT.log"
    
    visualize_bot(log_file)
