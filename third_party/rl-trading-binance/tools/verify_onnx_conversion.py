"""
Script to verify the correctness of ONNX model conversion by comparing outputs with original PyTorch models.
"""
import argparse
import sys
from pathlib import Path
import importlib.util
import torch
import onnxruntime as ort
import numpy as np

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
    parser = argparse.ArgumentParser(description="Verify ONNX model conversion.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the model's 'best.pth' and python config file."
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to the converted .onnx file."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numerical tolerance for comparing PyTorch and ONNX outputs."
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    onnx_path = Path(args.onnx_path)
    tolerance = args.tolerance

    if not model_dir.is_dir():
        print(f"❌ Error: Model directory not found at '{model_dir}'")
        sys.exit(1)
    if not onnx_path.exists():
        print(f"❌ Error: ONNX model not found at '{onnx_path}'")
        sys.exit(1)

    print(f"--- Verifying Model: {model_dir.name} ---")

    # 1. Load PyTorch model
    cfg_file = _find_config_file(model_dir)
    if not cfg_file:
        print(f"❌ Error: Python config file (*.py) not found in '{model_dir}'")
        sys.exit(1)
    cfg = _load_py_config(cfg_file)
    
    pytorch_agent = _create_agent_from_config(cfg)
    weights_path = model_dir / "best.pth"
    pytorch_agent.load_model(str(weights_path))
    pytorch_agent.policy_net.eval()
    print(f"✅ Loaded PyTorch model from '{weights_path}'")

    # 2. Load ONNX model
    onnx_session = ort.InferenceSession(str(onnx_path))
    onnx_input_name = onnx_session.get_inputs()[0].name
    print(f"✅ Loaded ONNX model from '{onnx_path}'")

    # 3. Create dummy input tensor (same as for export)
    input_shape = (1, 454) 
    dummy_input = torch.randn(input_shape, device=torch.device("cpu"))
    dummy_input_np = dummy_input.numpy()
    print(f"✅ Created dummy input tensor with shape {input_shape}")

    # 4. Get outputs
    with torch.no_grad():
        pytorch_output = pytorch_agent.policy_net(dummy_input).numpy()
    
    onnx_output = onnx_session.run(None, {onnx_input_name: dummy_input_np})[0]

    # 5. Compare outputs
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)

    if max_diff < tolerance:
        print(f"🎉 Verification SUCCESS! Maximum difference: {max_diff:.8f} (tolerance: {tolerance})")
    else:
        print(f"❌ Verification FAILED! Maximum difference: {max_diff:.8f} (tolerance: {tolerance})")
        print("PyTorch Output (first 5 values):", pytorch_output.flatten()[:5])
        print("ONNX Output (first 5 values):", onnx_output.flatten()[:5])
        sys.exit(1)

if __name__ == "__main__":
    main()
