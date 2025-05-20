# c:\Python\Prosperous_Bot\src\ml_model.py

import argparse
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import optuna
import logging
from adaptive_agent.pathlib import Path
import json
import os
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
from adaptive_agent.xgboost import XGBClassifier
from .signal_generator import generate_signal

CONFIG_KEY = 'ml_model'

class MLModel:
    """
    Обёртка над XGBoost и оптимизацией параметров фильтров.
    Читает локальные CSV-файлы OHLCV из папки graphs, вычисляет индикаторы,
    оптимизирует параметры фильтров (Optuna), обучает RandomForest,
    сохраняет модель, логи и визуализации (feature importance, signals HTML).
    Работает только с признаками EMA, SMA, Bollinger Bands и ADX из config_signal.json.
    """
    def __init__(self, config_path: str):
        # 1) загрузка конфига
        self.config = json.load(open(config_path, 'r', encoding='utf-8'))
        # 2) директория для ML-отчётов берём прямо из graph_dir
        self.ml_dir = Path(self.config['graph_dir'])
        self.ml_dir.mkdir(parents=True, exist_ok=True)
        # 3) единый логгер для оптимизации и обучения
        self.logger = logging.getLogger('ml_model')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.ml_dir / 'ml_model.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(fh)
        # остальные атрибуты
        self.config_path = config_path

    def _load_data(self, symbol: str) -> pd.DataFrame:
        """
        Загружает данные OHLCV из CSV, сохранённого signal_bot.
        Должен содержать столбец 'timestamp'.
        """
        # Корректно формируем имя файла: добавляем 'USDT' только если его нет
        sym = symbol.upper()
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        filename = f"{sym}_data.csv"
        path = Path('C:\\Python\\Prosperous_Bot\\graphs') / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет точно те же индикаторы, что и Signal Bot:
        - EMA1, EMA2, EMA3
        - MA1, MA2
        - Bollinger Bands (upper, lower, mid, width)
        - ADX
        Параметры берутся из конфигурации.
        """
        cfg = self.config
        # EMA
        df['ema1'] = ta.trend.EMAIndicator(df['close'], window=cfg['ema1_window']).ema_indicator()
        df['ema2'] = ta.trend.EMAIndicator(df['close'], window=cfg['ema2_window']).ema_indicator()
        df['ema3'] = ta.trend.EMAIndicator(df['close'], window=cfg['ema3_window']).ema_indicator()
        # MA
        df['ma1'] = ta.trend.SMAIndicator(df['close'], window=cfg['ma1_window']).sma_indicator()
        df['ma2'] = ta.trend.SMAIndicator(df['close'], window=cfg['ma2_window']).sma_indicator()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=cfg['bb_window'], window_dev=cfg.get('bb_std_dev', 2))
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        # абсолютная ширина
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        # относительная ширина (для фильтрации по волатильности)
        df['bb_width_rel'] = df['bb_width'] / df['bb_mid']
        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=cfg['adx_window']).adx()
        df = df.dropna()
        return df

    def prepare_dataset(self, symbol: str):
        """
        Формирует DataFrame с признаками и целевой переменной для обучения.
        Цель: бинарное направление цены через N свечей (target_shift).
        """
        df = self._load_data(symbol)
        df = self._compute_features(df)
        # Фильтрация по порогу волатильности из конфига
        vt = self.config.get('volatility_threshold', 0)
        if vt > 0:
            df = df[df['bb_width_rel'] > vt]
        # Целевая переменная: рост цены через target_shift свечей
        N = self.config['ml_model'].get('target_shift', 3)
        df['target'] = (df['close'].shift(-N) > df['close']).astype(int)
        df.dropna(subset=['target'], inplace=True)
        features = [
            'ema1', 'ema2', 'ema3', 'ma1', 'ma2',
            'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_width_rel', 'adx'
        ]
        X = df[features]
        y = df['target']
        test_size = self.config['ml_model'].get('test_size', 0.3)
        return train_test_split(X, y, shuffle=False, test_size=test_size)

    def optimize_filters(self, symbol: str, n_trials: int = 50) -> dict:
        """
        Optimize filter parameters using Optuna.

        Args:
        symbol: Trading pair symbol
        n_trials: Number of optimization trials

        Returns:
        dict: Best parameters found

        Raises:
        KeyError: If required config or columns missing
        """
        df = self._load_data(symbol)
        df = self._compute_features(df)
        # Уже не генерируем сигнал здесь — будем это делать ВНУТРИ objective,
        # чтобы trial-параметры влияли на сигналы.
        features = df.columns.intersection([
            'ema1', 'ema2', 'ema3', 'ma1', 'ma2', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_width_rel', 'adx'
        ])

        def objective(trial) -> float:
            """Для каждого trial: ставим trial-параметры, генерируем сигналы,
            разбиваем выборку и возвращаем f1_score."""
            # 1) Пробуем параметры из search_space
            ss = self.config['ml_model']['search_space']
            for p, (lo, hi) in ss.items():
                # Подбираем int-параметры целочисленно, а float-порог волатильности — с точностью
                if isinstance(lo, float) or isinstance(hi, float):
                    self.config[p] = trial.suggest_float(p, lo, hi)
                else:
                    self.config[p] = trial.suggest_int(p, lo, hi)
            # 2) Генерируем сигналы rule-based целиком с новыми параметрами
            data_sig, _, _ = generate_signal(df.copy(), self.config, last_signal=None)
            y = data_sig['Buy_Signal'].fillna(0).astype(int)
            # 3) Train-val split
            X_train, X_val, y_train, y_val = train_test_split(
                df[features], y,
                shuffle=False,
                test_size=self.config['ml_model'].get('test_size', 0.3)
            )
            # 4) Тренируем XGBoost (с корректным base_score и метриками)
            params = self.config['ml_model']['params'].copy()
            params.setdefault('base_score', 0.5)
            params.setdefault('use_label_encoder', False)
            params.setdefault('eval_metric', 'logloss')
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, zero_division=0)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        # логируем результаты по символу
        self.logger.info("Best params for %s: %s", symbol, best)
        # Save optimization results
        res_df = pd.DataFrame([best])
        res_df.to_csv(self.ml_dir / f"{symbol}_optimization_results.csv", index=False)
        logging.info(f"Optimization results saved: {symbol}_optimization_results.csv")
        return best

    def average_params(self, best_params_list: list):
        """Усредняет лист словарей best_params и обновляет config_signal.json."""
        # Усредняем параметры, сохраняя тип: int для целочисленных, float для вещественных
        avg = {}
        first = best_params_list[0]
        for k in first.keys():
            vals = [d[k] for d in best_params_list]
            mean_val = sum(vals) / len(vals)
            # Если исходные параметры были целыми, приводим к int, иначе оставляем float
            if isinstance(first[k], int):
                avg[k] = int(round(mean_val))
            else:
                avg[k] = mean_val
        # Запишем усреднённые параметры в верхний уровень конфига
        for p, v in avg.items():
            if p in self.config:
                self.config[p] = v
        # Сохраняем конфиг один раз с новыми усреднёнными значениями
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        logging.info("Averaged indicator windows saved to config: %s", avg)

    def plot_feature_importance(self, model, symbol: str):
        """Сохраняет PNG с важностью признаков."""
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance for {symbol}")
        plt.barh([model.feature_names_in_[i] for i in indices], importances[indices])
        plt.tight_layout()
        path = self.ml_dir / f"{symbol}_feature_importance.png"
        plt.savefig(str(path))
        plt.close()
        logging.info(f"Feature importance saved: {path}")

    def generate_signals_html(self, symbol: str, df: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray):
        """Генерирует интерактивный HTML с Candlestick + сигналами."""
        # df уже содержит timestamp, open, high, low, close
        n = len(df)
        split = int(n * (1 - self.config['ml_model'].get('test_size', 0.3)))
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]
        fig = go.Figure([go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price'
        )])
        # 1) ML-сигналы по порогу из конфига
        # Формируем X_test из тестовой части df по тем же признакам, что и при обучении
        features = [
            'ema1', 'ema2', 'ema3', 'ma1', 'ma2',
            'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_width_rel', 'adx'
        ]
        X_test = df_test[features]
        proba = self.model.predict_proba(X_test)[:, 1]
        thresh = self.config.get('threshold', 0.5)
        ml_buy = proba >= thresh
        ml_sell = proba < thresh

        # 2) Rule-based-сигналы по тем же индикаторам
        data_rb, _, _ = generate_signal(df_test.copy(), self.config, last_signal=None)
        rb_buy = data_rb['Buy_Signal'].fillna(0).astype(int) == 1
        rb_sell = data_rb['Sell_Signal'].fillna(0).astype(int) == 1

        # 3) Пересечение ML и rule-based
        buy = df_test[ml_buy & rb_buy]
        sell = df_test[ml_sell & rb_sell]

        fig.add_trace(go.Scatter(
            x=buy['timestamp'], y=buy['high'],
            mode='markers', marker_symbol='triangle-up', marker_color='green',
            marker_size=10, name='BUY (ML & RB)'
        ))
        fig.add_trace(go.Scatter(
            x=sell['timestamp'], y=sell['low'],
            mode='markers', marker_symbol='triangle-down', marker_color='red',
            marker_size=10, name='SELL (ML & RB)'
        ))
        # Заливка области train
        fig.add_vrect(
            x0=df_train['timestamp'].min(),
            x1=df_train['timestamp'].max(),
            fillcolor="LightGrey", opacity=0.2, layer="below", line_width=0
        )
        fig.update_xaxes(range=[df['timestamp'].min(), df['timestamp'].max()],
                         tickformat='%Y-%m-%d %H:%M')
        fig.update_layout(title=f"{symbol} Signals", xaxis_rangeslider_visible=False)
        out = self.ml_dir / f"{symbol}_signals.html"
        fig.write_html(str(out))

    def train(self, symbols: list):
        """Тренирует и сохраняет модель и HTML-сигналы для каждого символа."""
        metrics = []
        for symbol in symbols:
            # 1) Загружаем полный DataFrame с timestamp
            df = self._load_data(symbol)
            df = self._compute_features(df)
            # 2) Формируем target и отбрасываем NaN
            N = self.config['ml_model'].get('target_shift', 3)
            df['target'] = (df['close'].shift(-N) > df['close']).astype(int)
            df.dropna(subset=['target'], inplace=True)
            # 3) Делим на X и y
            features = [
                'ema1', 'ema2', 'ema3', 'ma1', 'ma2',
                'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_width_rel', 'adx'
            ]
            X = df[features]
            y = df['target']
            test_size = self.config['ml_model'].get('test_size', 0.3)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, shuffle=False, test_size=test_size
            )
            # 4) Тренируем модель XGBoost (с корректным base_score и метриками)
            params = self.config['ml_model']['params'].copy()
            params.setdefault('base_score', 0.5)
            params.setdefault('use_label_encoder', False)
            params.setdefault('eval_metric', 'logloss')
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # === КОРРЕКТНЫЕ Sharpe / MaxDD / Win-Rate ===
            # Рассчитываем только на тестовой выборке, чтобы избежать data-leak
            test_idx = X_test.index
            price_test = df.loc[test_idx, 'close']
            ret_test = price_test.pct_change().dropna()

            # Sharpe (365 торговых дня / 5-мин бар ≈ 12*24*365 ≈ 105 120 баров)
            ann_fact = np.sqrt(12*24*30)  # годовая/месячная шкала → хватит
            if ret_test.std() == 0:
                sharpe = 0.0
            else:
                sharpe = (ret_test.mean() / ret_test.std()) * ann_fact

            # Max Drawdown
            eq_curve = (1 + ret_test).cumprod()
            roll_max = eq_curve.cummax()
            max_dd = ((eq_curve - roll_max) / roll_max).min()  # отрицательное число

            # Win-Rate (доля правильных направлений)
            win_rate = (preds == y_test.values).mean()

            metrics.append({
                "symbol": symbol,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "win_rate": win_rate
            })

            # 5) Сохраняем модель и важность признаков
            raw = self.config['ml_model']['model_path'].format(symbol=symbol)
            model_file = self.ml_dir / Path(raw).name
            joblib.dump(model, str(model_file))
            self.plot_feature_importance(model, symbol)
            # 6) Перед отрисовкой сохраняем модель в self.model,
            # чтобы generate_signals_html мог взять predict_proba
            self.model = model
            # 7) Генерируем сигналы на тестовой части и HTML
            y_pred = model.predict(X_test)
            self.generate_signals_html(symbol, df, y_test, y_pred)
            print(f"Отчёты ml_model сохранены в: {self.ml_dir}")

        # в конце сохраняем metrics в summary_report.html
        import jinja2

        # ---- загружаем HTML-шаблон ----
        template_path = Path(__file__).resolve().parent.parent / "templates" / "summary_template.html"
        if template_path.exists():
            env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path.parent))
            tmpl = env.get_template(template_path.name)
            html_out = tmpl.render(rows=metrics)
        else:
            # fallback: простая таблица
            import pandas as pd
            html_out = pd.DataFrame(metrics).to_html(index=False)

        # ---- сохраняем отчёт ---- 
        REPORT_DIR = Path(r"C:\Python\Prosperous_Bot\graphs\ml")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "summary_report.html").write_text(html_out, encoding="utf8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize/train ML model")
    parser.add_argument("--config", "-c",
                        default=r"C:\Python\Prosperous_Bot\config_signal.json",
                        help="Path to config_signal.json")
    parser.add_argument("--symbols", "-s", nargs="+",
                        help="Symbols to process (e.g. BTC ETH)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run filter parameter optimization")
    parser.add_argument("--train", action="store_true",
                        help="Run model training")
    parser.add_argument("--trials", "-t", type=int, default=50,
                        help="Number of Optuna trials for optimization")
    args = parser.parse_args()

    # Load config
    try:
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
    except Exception as e:
        print(f"Cannot read config: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine symbols list
    symbols = args.symbols or cfg.get("binance_pairs", [])
    if not symbols:
        print("No symbols specified and binance_pairs empty in config", file=sys.stderr)
        sys.exit(1)

    # инициализация
    model = MLModel(args.config)
    best_list = []
    if args.optimize:
        for sym in symbols:
            best = model.optimize_filters(sym, n_trials=args.trials)
            best_list.append({'symbol': sym, **best})
        # 1) сохраняем агрегированные результаты в CSV
        import pandas as pd
        df_res = pd.DataFrame(best_list)
        csv_path = model.ml_dir / 'optimization_results.csv'
        df_res.to_csv(csv_path, index=False)
        # 2) сохраняем HTML-отчёт
        html = df_res.to_html(index=False)
        with open(model.ml_dir / 'summary_report.html', 'w', encoding='utf-8') as f:
            f.write(html)
        # усреднение и запись в конфиг
        model.average_params([ {k:v for k,v in row.items() if k!='symbol'} for row in best_list ])
    if args.train:
        model.train(symbols)