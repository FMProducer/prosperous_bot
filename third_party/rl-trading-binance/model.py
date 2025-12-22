import logging
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN with Dropout layers for estimating epistemic uncertainty.
    Uses 1D convolutions with dilations.
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
        self.additional_feats = additional_feats

        channels, history_len, _ = input_shape # width is ignored for Conv1d

        if cnn_dilations is None:
            cnn_dilations = [1] * len(cnn_kernels)

        cnn_layers = []
        in_channels = channels
        for out_ch, k, s, d in zip(cnn_maps, cnn_kernels, cnn_strides, cnn_dilations):
            # Causal padding for Conv1d
            padding = (k - 1) * d // 2
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=s,
                    dilation=d,
                    padding=padding,
                )
            )
            cnn_layers.append(nn.ReLU(inplace=True))
            cnn_layers.append(nn.Dropout(p=dropout_p))
            in_channels = out_ch
        self.feature_extractor = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            # Dummy input for Conv1d should be (batch, channels, seq_len)
            dummy = torch.zeros(1, channels, history_len)
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

        # Добавляем модули для квантования
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        logger.info(f"Initialized DuelingQNetwork with QAT support...")

    def forward(self, state: Tensor, return_components: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        # Оборачиваем вычисления для QAT
        state = self.quant(state)
        
        batch = state.size(0)

        history_flat_size = self.input_shape[0] * self.input_shape[1]
        expected_size = history_flat_size + self.additional_feats

        assert state.size(1) == expected_size, (
            f"Input state size mismatch. Expected {expected_size} "
            f"(history: {history_flat_size}, additional: {self.additional_feats}), "
            f"but got {state.size(1)}."
        )

        # The input state is flat, need to separate history and extra features
        history_part = state[:, :history_flat_size]
        extra_part = state[:, history_flat_size:]

        # Reshape for Conv1d: (batch, channels, seq_len)
        history_tensor = history_part.view(batch, self.input_shape[0], self.input_shape[1])

        features = self.feature_extractor(history_tensor)
        features_flat = features.view(batch, -1)

        combined = torch.cat([features_flat, extra_part], dim=1)

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)

        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        q_value = self.dequant(q_value)
        if return_components:
            return q_value, self.dequant(value), self.dequant(advantage)
        
        return q_value
