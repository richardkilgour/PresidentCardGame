from __future__ import annotations
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from president.models.PresidentEnv import PresidentEnv

MODEL_DIR = Path(__file__).parent
LOG_DIR   = MODEL_DIR / "rl_logs"


def mask_fn(env):
    return env.action_masks()


def warm_start(model: MaskablePPO) -> None:
    """
    Initialise the PPO policy's shared MLP layers and action head
    from the supervised clone weights.
    The value network is left random — it learns quickly once the
    policy is reasonable.
    """
    supervised_path = MODEL_DIR / "player_splitter_mlp.pt"
    if not supervised_path.exists():
        print("No supervised clone found — training from scratch.")
        return

    supervised = torch.load(supervised_path, weights_only=True)
    policy_state = model.policy.state_dict()

    # Shared MLP layers (two hidden layers of 256)
    policy_state["mlp_extractor.policy_net.0.weight"] = supervised["0.weight"]
    policy_state["mlp_extractor.policy_net.0.bias"]   = supervised["0.bias"]
    policy_state["mlp_extractor.policy_net.2.weight"] = supervised["2.weight"]
    policy_state["mlp_extractor.policy_net.2.bias"]   = supervised["2.bias"]

    # Action head (108 → 256 → 256 → 55)
    policy_state["action_net.weight"] = supervised["4.weight"]
    policy_state["action_net.bias"]   = supervised["4.bias"]

    model.policy.load_state_dict(policy_state)
    print("Warm-started from supervised clone.")


if __name__ == "__main__":
    env      = ActionMasker(PresidentEnv(), mask_fn)
    eval_env = ActionMasker(PresidentEnv(), mask_fn)

    best_model_path = MODEL_DIR / "best_model.zip"
    checkpoint_path = MODEL_DIR / "checkpoints"

    if best_model_path.exists():
        print(f"Resuming from {best_model_path}")
        model = MaskablePPO.load(
            best_model_path,
            env=env,
            tensorboard_log=str(LOG_DIR),
        )
    else:
        # Find the latest checkpoint if best_model doesn't exist
        checkpoints = sorted(checkpoint_path.glob("rl_agent_*_steps.zip"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Resuming from checkpoint {latest}")
            model = MaskablePPO.load(
                latest,
                env=env,
                tensorboard_log=str(LOG_DIR),
            )
        else:
            print("No saved model found — warm starting from supervised clone.")
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(LOG_DIR),
                policy_kwargs=dict(net_arch=[256, 256]),
            )
            warm_start(model)

    model.verbose = 1

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=10_000,
        n_eval_episodes=200,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_path),
        name_prefix="rl_agent",
    )

    model.learn(
        total_timesteps=5_000_000,
        callback=CallbackList([eval_callback, checkpoint_callback]),
        reset_num_timesteps=False,  # continue timestep count from where it left off
    )

    model.save(MODEL_DIR / "rl_agent")
    print(f"\nTraining complete. Model saved to {MODEL_DIR / 'rl_agent'}")
