#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO


class SB3PolicyWrapper(torch.nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        self.policy = sb3_policy

    def forward(self, x):
        # x: [B, 7], output: [B, 2]
        return self.policy._predict(x, deterministic=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to SB3 PPO .zip checkpoint")
    parser.add_argument("--out", required=True, help="Output TorchScript .pt path")
    parser.add_argument("--obs-spec", default="", help="Path to obs_spec.json (recommended)")
    parser.add_argument("--obs-dim", type=int, default=0, help="Override model input dimension")
    args = parser.parse_args()

    ckpt = Path(args.ckpt).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(ckpt), device="cpu")
    model.policy.eval()

    obs_dim = int(args.obs_dim)
    if obs_dim <= 0 and args.obs_spec:
        spec_path = Path(args.obs_spec).resolve()
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        obs_dim = int(spec.get("obs_dim", 0))
    if obs_dim <= 0:
        obs_dim = 7

    wrapper = SB3PolicyWrapper(model.policy).eval()
    example = torch.zeros((1, obs_dim), dtype=torch.float32)
    ts = torch.jit.trace(wrapper, example)
    ts.save(str(out))
    print(f"Saved TorchScript model: {out} (obs_dim={obs_dim})")


if __name__ == "__main__":
    main()
