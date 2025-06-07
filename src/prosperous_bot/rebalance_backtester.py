def simulate_rebalance(data, orders_by_step, leverage=5.0):
    import logging
    open_positions = {}
    trade_log = []

    for idx, row in data.iterrows():
        price = row["close"]
        orders = orders_by_step.get(idx, [])
        timestamp = row["timestamp"] if "timestamp" in row else str(idx)

        for order in orders:
            key = order["asset_key"]
            side = order["side"]
            qty = order["qty"]
            if qty == 0:
                continue

            pos = open_positions.get(key)
            direction = 1 if side == "buy" else -1

            if pos is None:
                open_positions[key] = {
                    "entry_price": price,
                    "qty": qty,
                    "direction": direction,
                    "entry_time": timestamp,
                }
                continue

            same_dir = pos["direction"] == direction
            if same_dir:
                total_qty = pos["qty"] + qty
                avg_price = (pos["entry_price"] * pos["qty"] + price * qty) / total_qty
                pos["entry_price"] = avg_price
                pos["qty"] = total_qty
            else:
                closing_qty = min(pos["qty"], qty)
                pnl = (price - pos["entry_price"]) * closing_qty * pos["direction"] * leverage
                trade_log.append({
                    "asset_key": key,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "qty": closing_qty,
                    "pnl_gross_quote": pnl,
                    "leverage": leverage
                })

                if closing_qty < pos["qty"]:
                    pos["qty"] -= closing_qty
                elif closing_qty > pos["qty"]:
                    open_positions[key] = {
                        "entry_price": price,
                        "qty": qty - closing_qty,
                        "direction": direction,
                        "entry_time": timestamp,
                    }
                else:
                    del open_positions[key]

    logging.info(f"[simulate_rebalance] Завершено. Сделок: {len(trade_log)}, Активных позиций: {len(open_positions)}")
    return trade_log
