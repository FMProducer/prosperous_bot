"""
baseline_cnn_classifier.py — Baseline #2: Convolutional Classifier.
Trains a CNN on price history to predict direction + PnL/WinRate.
"""


import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from utils import millify
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from config import MasterConfig
from config import cfg as default_cfg
from utils import (
    apply_normalization,
    calculate_normalization_stats,
    compute_metrics,
    load_config,
    load_npz_dataset,
    select_and_arrange_channels,
    set_random_seed,
    setup_logging,
)


class CNNBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        cnn_maps: List[int],
        cnn_kernels: List[int],
        cnn_strides: List[int],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        channels, history_len, width = input_shape
        conv_layers = []
        in_ch = channels
        for out_ch, kernels, strides in zip(cnn_maps, cnn_kernels, cnn_strides):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(kernels, width), stride=(strides, 1)))
            conv_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            cnn_out = self.feature_extractor(dummy)
            flat_cnn_size = cnn_out.view(1, -1).size(1)

        self.classifier = nn.Linear(flat_cnn_size, 1)

    def forward(self, state: Tensor) -> Tensor:
        batch = state.size(0)
        features = self.feature_extractor(state)
        features = features.view(batch, -1)
        return torch.sigmoid(self.classifier(features)).squeeze(1)


class PriceDataset(Dataset):
    def __init__(self, sessions: List[np.ndarray], raw_prices: List[np.ndarray], cfg: MasterConfig):
        self.sessions = sessions
        self.raw_prices = raw_prices
        self.cfg = cfg
        self.close_idx = cfg.data.data_channels.index("close")

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        session = self.sessions[idx]
        prices = self.raw_prices[idx]
        start_price = prices[self.cfg.seq.pre_signal_len - 1, self.close_idx]
        end_price = prices[self.cfg.seq.pre_signal_len + self.cfg.seq.agent_session_len - 1, self.close_idx]
        label = 1 if (end_price - start_price) / start_price > 0 else 0
        return torch.tensor(session.T[:, :, None], dtype=torch.float32), label


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    raw_val: List[np.ndarray],
    cfg: MasterConfig,
) -> nn.Module:
    if cfg.device.device.type == "cpu":
        torch.set_flush_denormal(True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.rl.learning_rate)
    model.to(cfg.device.device)

    best_model_state = None
    best_val_pnl = float("-inf")

    logging.info(f"One epoch is {len(train_loader):.1f} iterations")

    model.train()
    train_iter = iter(train_loader)
    counter = trange(1, cfg.trainlog.iterations + 1, desc="Training", leave=False)
    smooth_loss = None
    for batch_idx in counter:
        try:
            x_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, y_batch = next(train_iter)

        x_batch, y_batch = x_batch.to(cfg.device.device), y_batch.float().to(cfg.device.device)
        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        if cfg.rl.max_gradient_norm is not None:
            clip_grad_norm_(model.parameters(), cfg.rl.max_gradient_norm)

        optimizer.step()

        smooth_loss = (smooth_loss or loss.item()) * 0.9 + loss.item() * 0.1
        counter.desc = f"Training loss={smooth_loss:.7f}"

        if (batch_idx + 1) % cfg.trainlog.val_freq == 0:
            model.eval()
            val_preds = []
            val_prob_preds = []
            val_targets = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(cfg.device.device)
                    pred = model(x_val).cpu().numpy()
                    val_preds.extend((pred > 0.5).astype(int))
                    val_prob_preds.extend(pred)
                    val_targets.extend(y_val)

            val_pnl, val_win = compute_metrics(raw_val, val_preds, cfg)
            roc_auc = roc_auc_score(val_targets, val_prob_preds)
            logging.info(
                f"Val results on iteration {batch_idx+1}: Mean PnL={val_pnl:.2f}, Win Rate={val_win:.2%}, ROC AUC={roc_auc:.2%}"
            )

            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_model_state = model.state_dict()
                best_model = (
                    f"Load Best model on Val from iteration {batch_idx+1}: Mean PnL={val_pnl:.2f}, "
                    f"Win Rate={val_win:.2%}, ROC AUC={roc_auc:.2%}"
                )
            model.train()
    logging.info(best_model)
    model.load_state_dict(best_model_state)
    return model


def run_baseline_cnn(cfg: MasterConfig):
    timestamp = time.strftime("date_%Y%m%d_time_%H%M%S")
    session_name = f"{cfg.project_name}_baseline_cnn_{timestamp}"
    setup_logging(session_name, cfg)
    set_random_seed(cfg.random_seed)
    os.makedirs(cfg.paths.plot_dir, exist_ok=True)

    raw_train = load_npz_dataset(
        cfg.paths.train_data_path,
        "Train",
        cfg.paths.plot_dir,
        cfg.debug.debug_max_size_data,
        cfg.data.plot_examples,
        cfg.data.plot_channel_idx,
        cfg.seq.pre_signal_len,
    )

    raw_val = load_npz_dataset(
        cfg.paths.val_data_path,
        "Val",
        cfg.paths.plot_dir,
        cfg.debug.debug_max_size_data,
        cfg.data.plot_examples,
        cfg.data.plot_channel_idx,
        cfg.seq.pre_signal_len,
    )

    raw_test = load_npz_dataset(
        cfg.paths.test_data_path,
        "Test",
        cfg.paths.plot_dir,
        cfg.debug.debug_max_size_data,
        cfg.data.plot_examples,
        cfg.data.plot_channel_idx,
        cfg.seq.pre_signal_len,
    )

    raw_train = [
        select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels) for _, arr in raw_train
    ]
    raw_val = [
        select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels) for _, arr in raw_val
    ]
    raw_test = [
        select_and_arrange_channels(arr, cfg.data.expected_channels, cfg.data.data_channels) for _, arr in raw_test
    ]

    stats = calculate_normalization_stats(
        raw_train, cfg.data.data_channels, cfg.data.price_channels, cfg.data.volume_channels, cfg.data.other_channels
    )

    train_seqs = [
        apply_normalization(
            seq[cfg.seq.pre_signal_len - cfg.seq.agent_history_len : cfg.seq.pre_signal_len],
            stats,
            cfg.data.data_channels,
            cfg.data.price_channels,
            cfg.data.volume_channels,
            cfg.data.other_channels,
            cfg.seq.agent_history_len,
            cfg.seq.input_history_len,
        )
        for seq in raw_train
    ]

    val_seqs = [
        apply_normalization(
            seq[cfg.seq.pre_signal_len - cfg.seq.agent_history_len : cfg.seq.pre_signal_len],
            stats,
            cfg.data.data_channels,
            cfg.data.price_channels,
            cfg.data.volume_channels,
            cfg.data.other_channels,
            cfg.seq.agent_history_len,
            cfg.seq.input_history_len,
        )
        for seq in raw_val
    ]

    test_seqs = [
        apply_normalization(
            seq[cfg.seq.pre_signal_len - cfg.seq.agent_history_len : cfg.seq.pre_signal_len],
            stats,
            cfg.data.data_channels,
            cfg.data.price_channels,
            cfg.data.volume_channels,
            cfg.data.other_channels,
            cfg.seq.agent_history_len,
            cfg.seq.input_history_len,
        )
        for seq in raw_test
    ]

    train_ds = PriceDataset(train_seqs, raw_train, cfg)
    val_ds = PriceDataset(val_seqs, raw_val, cfg)
    test_ds = PriceDataset(test_seqs, raw_test, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.rl.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.rl.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.rl.batch_size, shuffle=False)

    input_shape = (cfg.seq.num_features, cfg.seq.input_history_len, 1)
    model = CNNBinaryClassifier(
        input_shape=input_shape,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"CNNBinaryClassifier with {millify(num_params, precision=1)} parameters created.")

    model = train_model(model, train_loader, val_loader, raw_val, cfg)

    model.eval()
    all_preds = []
    test_prob_preds = []
    test_targets = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(cfg.device.device)
            pred = model(x_test).cpu().numpy()
            all_preds.extend((pred > 0.5).astype(int))
            test_prob_preds.extend(pred)
            test_targets.extend(y_test)

    mean_pnl, win_rate = compute_metrics(raw_test, all_preds, cfg)
    roc_auc = roc_auc_score(test_targets, test_prob_preds)

    logging.info("\n* CNN Classifier *")
    logging.info(f"Number of sessions: {len(test_ds)}")
    logging.info(
        f"--- Test CNN Classifier Results: Mean PnL = {mean_pnl:.2f}, Win Rate = {win_rate:.2%}, ROC AUC = {roc_auc:.2%} ---"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg = load_config(sys.argv[1])
    else:
        cfg = default_cfg

    run_baseline_cnn(cfg)
