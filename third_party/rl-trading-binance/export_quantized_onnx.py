
import argparse
import logging
import os
import sys

import torch

from agent import D3QN_PER_Agent
from utils import load_config

logger = logging.getLogger(__name__)

def main():
    """
    Loads a trained D3QN_PER_Agent, applies dynamic quantization,
    and exports the policy network to an INT8 ONNX model.
    """
    parser = argparse.ArgumentParser(
        description="Export a trained model to a quantized INT8 ONNX format."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the quantized ONNX model.",
    )
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    cfg, _ = load_config(args.config, return_module=True)

    # Initialize agent
    agent = D3QN_PER_Agent(
        state_shape=(cfg.seq.num_features, cfg.seq.input_history_len, 1),
        action_dim=cfg.market.num_actions,
        config=cfg,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
        cnn_dilations=getattr(cfg.model, 'cnn_dilations', None),
        dense_val=cfg.model.dense_val,
        dense_adv=cfg.model.dense_adv,
        additional_feats=cfg.model.additional_feats,
        dropout_model=cfg.model.dropout_p,
        device=torch.device("cpu"),  # Quantization is for CPU
        gamma=cfg.rl.gamma,
        learning_rate=cfg.rl.lr,
        batch_size=cfg.rl.batch_size,
        buffer_size=cfg.per.buffer_size,
        target_update_freq=cfg.rl.target_update_freq,
        train_start=cfg.rl.train_start,
        per_alpha=cfg.per.per_alpha,
        per_beta_start=cfg.per.per_beta_start,
        per_beta_frames=cfg.per.per_beta_frames,
        eps_start=cfg.eps.eps_start,
        eps_end=cfg.eps.eps_end,
        eps_frames=cfg.eps.eps_decay_frames,
        epsilon=cfg.per.per_eps,
        max_gradient_norm=cfg.rl.max_gradient_norm,
    )

    # Load the trained model checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Model checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    agent.load_model(args.checkpoint)
    logger.info(f"Successfully loaded model from {args.checkpoint}")

    # Apply dynamic quantization
    agent.policy_net = agent.policy_net.prepare_for_cpu_inference()
    logger.info("Model has been quantized for CPU inference.")

    # Export the quantized model to ONNX
    agent.export_onnx(args.output)
    logger.info(f"Quantized model exported to {args.output}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
