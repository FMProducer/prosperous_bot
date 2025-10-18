import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN with Dropout layers for estimating epistemic uncertainty.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        action_dim: int,
        cnn_maps: List[int],
        cnn_kernels: List[int],
        cnn_strides: List[int],
        dense_val: List[int],
        dense_adv: List[int],
        additional_feats: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim

        channels, history_len, width = input_shape
        conv_layers = []
        in_ch = channels
        for out_ch, kernels, strides in zip(cnn_maps, cnn_kernels, cnn_strides):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(kernels, width), stride=(strides, 1)))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(p=dropout_p))
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            cnn_out = self.feature_extractor(dummy)
            flat_cnn_size = cnn_out.view(1, -1).size(1)

        mlp_input_size = flat_cnn_size + additional_feats

        # Value stream
        value_layers = []
        prev = mlp_input_size
        for units in dense_val:
            value_layers.extend([nn.Linear(prev, units), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p)])
            prev = units
        value_layers.append(nn.Linear(prev, 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Advantage stream
        adv_layers = []
        prev = mlp_input_size
        for units in dense_adv:
            adv_layers.extend([nn.Linear(prev, units), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p)])
            prev = units
        adv_layers.append(nn.Linear(prev, action_dim))
        self.advantage_stream = nn.Sequential(*adv_layers)

        logger.info(f"Initialized DuelingQNetwork: input={input_shape}, actions={action_dim}")

    def forward(self, state: Tensor) -> Tensor:
        batch = state.size(0)
        history_flat_size = torch.prod(torch.tensor(self.input_shape)).item()
        history_part = state[:, :history_flat_size]
        extra_part = state[:, history_flat_size:]

        history_tensor = history_part.view(batch, *self.input_shape)

        features = self.feature_extractor(history_tensor)
        features_flat = features.view(batch, -1)

        combined = torch.cat([features_flat, extra_part], dim=1)

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value
