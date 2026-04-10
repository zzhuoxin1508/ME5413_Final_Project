#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime
from collections import defaultdict

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.nav2d_env import Nav2DEnv


def make_env(cfg):
    def _thunk():
        return Monitor(
            Nav2DEnv(cfg),
            info_keywords=(
                "ep_r_progress",
                "ep_r_step",
                "ep_r_align",
                "ep_r_speed",
                "ep_r_safe",
                "ep_r_spin",
                "ep_r_stall",
                "ep_r_collision",
                "ep_r_goal",
                "ep_r_heading",
                "ep_r_turn",
                "ep_r_total",
            ),
        )
    return _thunk


class RewardBreakdownCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_acc = defaultdict(list)
        self.ep_acc = defaultdict(list)
        self.step_keys = (
            "r_progress",
            "r_step",
            "r_align",
            "r_speed",
            "r_safe",
            "r_spin",
            "r_stall",
            "r_collision",
            "r_goal",
            "r_heading",
            "r_turn",
            "r_total",
        )
        self.ep_keys = (
            "ep_r_progress",
            "ep_r_step",
            "ep_r_align",
            "ep_r_speed",
            "ep_r_safe",
            "ep_r_spin",
            "ep_r_stall",
            "ep_r_collision",
            "ep_r_goal",
            "ep_r_heading",
            "ep_r_turn",
            "ep_r_total",
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for k in self.step_keys:
                if k in info:
                    self.step_acc[k].append(float(info[k]))
            for k in self.ep_keys:
                if k in info:
                    self.ep_acc[k].append(float(info[k]))
        return True

    def _on_rollout_end(self) -> None:
        for k, values in self.step_acc.items():
            if values:
                self.logger.record(f"reward_components/step_mean/{k}", sum(values) / len(values))
        for k, values in self.ep_acc.items():
            if values:
                self.logger.record(f"reward_components/episode_mean/{k}", sum(values) / len(values))
        self.step_acc.clear()
        self.ep_acc.clear()


def build_wandb_callback(train_cfg, cfg, save_dir, cli_enable):
    wandb_cfg = train_cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False) or cli_enable)
    if not enabled:
        return None, None

    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except Exception as exc:
        raise RuntimeError(
            "W&B is enabled but import failed. Please run: pip install wandb"
        ) from exc

    run_name = wandb_cfg.get("run_name", "").strip()
    if not run_name:
        run_name = f"{train_cfg.get('model_name', 'ppo_nav2d')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    tags = wandb_cfg.get("tags", [])
    entity = wandb_cfg.get("entity", "").strip() or None
    project = wandb_cfg.get("project", "me5413-nav2d")

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=cfg,
        tags=tags,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    wb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=str(save_dir / "wandb_models"),
        verbose=1,
    )
    return run, wb_cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to nav_rl.yaml")
    parser.add_argument("--wandb", action="store_true", help="Force enable W&B logging")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    obs_cfg = cfg.get("obs", {})
    save_dir = Path(train_cfg["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(cfg)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(train_cfg["learning_rate"]),
        n_steps=int(train_cfg["n_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
        ent_coef=float(train_cfg["ent_coef"]),
        vf_coef=float(train_cfg["vf_coef"]),
        clip_range=float(train_cfg["clip_range"]),
        seed=int(train_cfg["seed"]),
        verbose=1,
        tensorboard_log=str(save_dir / "tb"),
    )
    wb_run, wb_cb = build_wandb_callback(train_cfg, cfg, save_dir, args.wandb)
    reward_cb = RewardBreakdownCallback()
    callbacks = [reward_cb]
    if wb_cb is not None:
        callbacks.append(wb_cb)
    callback = CallbackList(callbacks)
    model.learn(total_timesteps=int(train_cfg["total_steps"]), callback=callback)

    model_name = train_cfg.get("model_name", "ppo_nav2d")
    out_ckpt = save_dir / f"{model_name}_final.zip"
    model.save(str(out_ckpt))
    print(f"Saved checkpoint: {out_ckpt}")

    # Save observation spec for deployment alignment
    obs_spec = {
        "obs_order": ["dist_goal", "yaw_err", "front", "left", "right", "v_curr", "w_curr"],
        "frame_stack": int(obs_cfg.get("frame_stack", 1)),
        "obs_dim": int(len(["dist_goal", "yaw_err", "front", "left", "right", "v_curr", "w_curr"]) * int(obs_cfg.get("frame_stack", 1))),
        "act_order": ["linear_x", "angular_z"],
    }
    with open(save_dir / "obs_spec.json", "w", encoding="utf-8") as f:
        json.dump(obs_spec, f, indent=2)
    print(f"Saved obs spec: {save_dir / 'obs_spec.json'}")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
