"""
Script to convert a PyTorch model used in the D3QN strategy to the ONNX format.
"""
import argparse
import sys
from pathlib import Path
import importlib.util
import torch

# --- Add project root to sys.path to allow importing 'agent' ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from agent import D3QN_PER_Agent
except ImportError as e:
    print(f"❌ CRITICAL: Could not import D3QN_PER_Agent. Ensure 'agent.py' is in the project root: {project_root}")
    sys.exit(1)

def _find_config_file(dir_path: Path):
    """Finds the model's Python configuration file (e.g., alpha_...py)."""
    for file in dir_path.glob("*.py"):
        if "alpha" in file.name or "config" in file.name:
            return file
    return None

def _load_py_config(file_path: Path):
    """Loads a Python module from a given path."""
    spec = importlib.util.spec_from_file_location("mod_cfg", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cfg

def _create_agent_from_config(cfg):
    """Instantiates a D3QN_PER_Agent from a loaded config module."""
    # This function is a simplified version of the one in the strategy
    agent = D3QN_PER_Agent(
        state_shape=cfg.seq.state_shape,
        action_dim=cfg.market.num_actions,
        cnn_maps=cfg.model.cnn_maps,
        cnn_kernels=cfg.model.cnn_kernels,
        cnn_strides=cfg.model.cnn_strides,
        cnn_dilations=cfg.model.cnn_dilations,
        dense_val=cfg.model.dense_val,
        dense_adv=cfg.model.dense_adv,
        additional_feats=cfg.model.additional_feats,
        dropout_model=cfg.model.dropout_p,
        device=torch.device("cpu"),
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
        epsilon=0.0,
        max_gradient_norm=cfg.rl.max_gradient_norm
    )
    return agent

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch D3QN model to ONNX.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the model's 'best.pth' and python config file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output .onnx file."
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_file = Path(args.output_file)

    if not model_dir.is_dir():
        print(f"❌ Error: Model directory not found at '{model_dir}'")
        sys.exit(1)

    # 1. Find and load model config
    cfg_file = _find_config_file(model_dir)
    if not cfg_file:
        print(f"❌ Error: Python config file (*.py) not found in '{model_dir}'")
        sys.exit(1)
    
    print(f"✅ Found config file: {cfg_file.name}")
    cfg = _load_py_config(cfg_file)

    # 2. Create agent and load weights
    agent = _create_agent_from_config(cfg)
    weights_path = model_dir / "best.pth"
    if not weights_path.exists():
        print(f"❌ Error: 'best.pth' not found in '{model_dir}'")
        sys.exit(1)

    agent.load_model(str(weights_path))
    agent.policy_net.eval()
    print(f"✅ Loaded weights from 'best.pth'")

    # 3. Create dummy input tensor
    # Based on get_model_input in the strategy: (window * channels + additional_features)
    # 90 * 5 + 4 = 454
    input_shape = (1, 454) 
    dummy_input = torch.randn(input_shape, device=torch.device("cpu"))
    print(f"✅ Created dummy input tensor with shape {input_shape}")

    # 4. Export to ONNX
    try:
        torch.onnx.export(
            agent.policy_net,
            dummy_input,
            str(output_file),
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        print(f"🎉 Successfully exported model to '{output_file}'")
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
