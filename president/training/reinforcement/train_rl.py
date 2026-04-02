#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reinforcement learning training for the splitter agent.

Usage:
    python train_rl.py <experiment_name> [--clone <clone_experiment_name>]

Reads RL config from:
    training/experiments/<experiment_name>/config.json

Warm-starts from supervised clone (if --clone is given, or "clone" key in config):
    training/experiments/<clone_experiment_name>/player_splitter_mlp.pt
    training/experiments/<clone_experiment_name>/config_checkpoint.json

Writes to:
    training/experiments/<experiment_name>/
        best_model.zip
        rl_agent.zip
        checkpoints/
        rl_logs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from president.training.reinforcement.PresidentEnv import PresidentEnv

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"

DEFAULT_RL_CONFIG = {
    "total_timesteps": 5_000_000,
    "eval_freq":       10_000,
    "n_eval_episodes": 200,
    "checkpoint_freq": 50_000,
    "net_arch":        [256, 256],
}


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_rl_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {exp_dir}")
    with open(config_path) as f:
        cfg = json.load(f)
    rl_cfg = cfg.get("rl", {})
    for key, value in DEFAULT_RL_CONFIG.items():
        rl_cfg.setdefault(key, value)
    return rl_cfg


# ─────────────────────────────────────────────
# Warm start
# ─────────────────────────────────────────────

def warm_start(model: MaskablePPO, clone_exp_dir: Path) -> None:
    """
    Initialise the PPO policy's shared MLP layers and action head
    from the supervised clone weights.
    The value network is left random — it learns quickly once the
    policy is reasonable.
    """
    checkpoint = clone_exp_dir / f"{clone_exp_dir}.pt"
    config_path = clone_exp_dir / "config_checkpoint.json"

    if not checkpoint.exists():
        print(f"No supervised clone found at {checkpoint} — training from scratch.")
        return
    if not config_path.exists():
        sys.exit(f"config_checkpoint.json not found in {clone_exp_dir} — cannot safely map weights.")

    with open(config_path) as f:
        clone_cfg = json.load(f)

    clone_arch = clone_cfg.get("hidden_sizes", [])
    rl_arch    = model.policy.mlp_extractor.policy_net
    rl_layers  = [m for m in rl_arch if isinstance(m, torch.nn.Linear)]

    if len(clone_arch) != len(rl_layers):
        sys.exit(
            f"Architecture mismatch: clone has {len(clone_arch)} hidden layers, "
            f"RL policy has {len(rl_layers)}. Adjust net_arch in the RL config."
        )

    supervised   = torch.load(checkpoint, weights_only=True, map_location="cpu")
    policy_state = model.policy.state_dict()

    # Map hidden layers: clone layer i lives at index 2i in the Sequential
    # (Linear at 0, ReLU at 1, Linear at 2, ReLU at 3, ...)
    for i in range(len(clone_arch)):
        src = f"{i * 2}"   # supervised Sequential index
        dst = f"mlp_extractor.policy_net.{i * 2}"
        policy_state[f"{dst}.weight"] = supervised[f"{src}.weight"]
        policy_state[f"{dst}.bias"]   = supervised[f"{src}.bias"]

    # Action head: last layer of supervised network
    last_idx = len(clone_arch) * 2
    policy_state["action_net.weight"] = supervised[f"{last_idx}.weight"]
    policy_state["action_net.bias"]   = supervised[f"{last_idx}.bias"]

    model.policy.load_state_dict(policy_state)
    print(f"Warm-started from supervised clone: {checkpoint}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def mask_fn(env):
    return env.action_masks()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL training for the splitter agent.")
    parser.add_argument("experiment", help="RL experiment name under training/experiments/")
    parser.add_argument("--clone", default=None,
                        help="Supervised clone experiment to warm-start from. "
                             "Falls back to 'clone' key in config.json if not given.")
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.experiment
    if not exp_dir.is_dir():
        sys.exit(f"Experiment directory not found: {exp_dir}")

    rl_cfg = load_rl_config(exp_dir)
    print(f"Experiment : {args.experiment}")
    print(f"RL config  : {json.dumps(rl_cfg, indent=2)}\n")

    # Resolve clone experiment
    clone_name = args.clone or rl_cfg.get("clone")
    clone_exp_dir = EXPERIMENTS_DIR / clone_name if clone_name else None

    log_dir        = exp_dir / "rl_logs"
    checkpoint_dir = exp_dir / "checkpoints"
    best_model_path = exp_dir / "best_model.zip"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    env      = ActionMasker(PresidentEnv(), mask_fn)
    eval_env = ActionMasker(PresidentEnv(), mask_fn)

    # ── Load or create model ──────────────────
    if best_model_path.exists():
        print(f"Resuming from {best_model_path}")
        model = MaskablePPO.load(
            best_model_path,
            env=env,
            tensorboard_log=str(log_dir),
        )
    else:
        checkpoints = sorted(checkpoint_dir.glob("rl_agent_*_steps.zip"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Resuming from checkpoint {latest}")
            model = MaskablePPO.load(
                latest,
                env=env,
                tensorboard_log=str(log_dir),
            )
        else:
            print("No saved model found — creating new model.")
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(log_dir),
                policy_kwargs=dict(net_arch=rl_cfg["net_arch"]),
            )
            if clone_exp_dir is not None:
                warm_start(model, clone_exp_dir)
            else:
                print("No clone specified — training from scratch.")

    model.verbose = 1

    # ── Callbacks ─────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(exp_dir),
        log_path=str(log_dir),
        eval_freq=rl_cfg["eval_freq"],
        n_eval_episodes=rl_cfg["n_eval_episodes"],
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=rl_cfg["checkpoint_freq"],
        save_path=str(checkpoint_dir),
        name_prefix="rl_agent",
    )

    model.learn(
        total_timesteps=rl_cfg["total_timesteps"],
        callback=CallbackList([eval_callback, checkpoint_callback]),
        reset_num_timesteps=False,
    )

    final_path = exp_dir / "rl_agent.zip"
    model.save(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
