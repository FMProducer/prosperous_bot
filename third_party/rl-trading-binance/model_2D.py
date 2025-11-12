import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN with Dropout layers for estimating epistemic uncertainty.
    Uses 2D convolutions.
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
        cnn_dilations: Optional[List[int]] = None,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim

        channels, history_len, width = input_shape

        if cnn_dilations is None:
            cnn_dilations = [1] * len(cnn_kernels)

        cnn_layers = []
        in_channels = channels
        for out_ch, k, s, d in zip(cnn_maps, cnn_kernels, cnn_strides, cnn_dilations):
            # Padding for Conv2d. We assume kernel is (k, 1) and we only pad height.
            padding = 0
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=(k, 1), # Kernel is (height, width)
                    stride=(s, 1),      # Stride is (height, width)
                    dilation=(d, 1),    # Dilation is (height, width)
                    padding=padding,
                )
            )
            cnn_layers.append(nn.ReLU(inplace=True))
            cnn_layers.append(nn.Dropout(p=dropout_p))
            in_channels = out_ch
        self.feature_extractor = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            # Dummy input for Conv2d should be (batch, channels, height, width)
            dummy = torch.zeros(1, channels, history_len, width)
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

        logger.info(f"Initialized DuelingQNetwork (Conv2d): input=(C:{channels}, H:{history_len}, W:{width}), actions={action_dim}")

    def forward(self, state: Tensor) -> Tensor:
        batch = state.size(0)
        # The input state is flat, need to separate history and extra features
        history_flat_size = self.input_shape[0] * self.input_shape[1] * self.input_shape[2] # C * H * W
        history_part = state[:, :history_flat_size]
        extra_part = state[:, history_flat_size:]

        # Reshape for Conv2d: (batch, channels, height, width)
        history_tensor = history_part.view(batch, self.input_shape[0], self.input_shape[1], self.input_shape[2])

        features = self.feature_extractor(history_tensor)
        features_flat = features.view(batch, -1)

        combined = torch.cat([features_flat, extra_part], dim=1)

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value
