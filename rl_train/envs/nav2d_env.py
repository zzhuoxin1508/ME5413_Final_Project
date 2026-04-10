#!/usr/bin/env python3
import math
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from PIL import Image
from gymnasium import spaces


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class Nav2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        env = cfg["env"]
        robot = cfg["robot"]
        reward = cfg["reward"]
        obs_cfg = cfg.get("obs", {})

        self.dt = float(env["dt"])
        self.max_steps = int(env["max_steps"])
        self.goal_tolerance = float(env["goal_tolerance"])
        self.collision_radius = float(env["collision_radius"])
        self.min_start_goal_dist = float(env["min_start_goal_dist"])
        self.max_start_goal_dist = float(env["max_start_goal_dist"])
        self.lidar_max_range = float(env["lidar_max_range"])
        self.lidar_front_deg = float(env["lidar_front_deg"])
        self.ray_step = float(env["ray_step"])

        self.max_lin = float(robot["max_lin"])
        self.max_ang = float(robot["max_ang"])

        self.progress_weight = float(reward["progress_weight"])
        self.step_penalty = float(reward["step_penalty"])
        self.collision_penalty = float(reward["collision_penalty"])
        self.goal_reward = float(reward["goal_reward"])
        self.heading_weight = float(reward.get("heading_weight", 0.5))
        self.speed_weight = float(reward.get("speed_weight", 0.1))
        self.safe_weight = float(reward.get("safe_weight", 0.05))
        self.turn_weight = float(reward.get("turn_weight", 0.1))
        self.stall_penalty = float(reward.get("stall_penalty", -0.05))
        self.stall_progress_eps = float(reward.get("stall_progress_eps", 1e-3))
        self.spin_stop_v_thresh = float(reward.get("spin_stop_v_thresh", 0.05))

        self.obs_keys = ["dist_goal", "yaw_err", "front", "left", "right", "v_curr", "w_curr"]
        self.base_obs_dim = len(self.obs_keys)
        self.frame_stack = int(obs_cfg.get("frame_stack", 1))
        self.obs_dim = self.base_obs_dim * self.frame_stack
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.max_lin, -self.max_ang], dtype=np.float32),
            high=np.array([self.max_lin, self.max_ang], dtype=np.float32),
            dtype=np.float32,
        )

        self._load_map(Path(env["map_yaml"]))
        self.rng = random.Random()
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.prev_dist_goal = 0.0
        self.obs_history = deque(maxlen=self.frame_stack)
        self.ep_r_progress = 0.0
        self.ep_r_step = 0.0
        self.ep_r_align = 0.0
        self.ep_r_speed = 0.0
        self.ep_r_safe = 0.0
        self.ep_r_spin = 0.0
        self.ep_r_stall = 0.0
        self.ep_r_collision = 0.0
        self.ep_r_goal = 0.0

    def _load_map(self, yaml_path: Path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            mcfg = yaml.safe_load(f)
        img_path = yaml_path.parent / mcfg["image"]
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        # 0=occupied, 255=free for this project
        self.occ = arr < 128
        self.height, self.width = self.occ.shape
        self.resolution = float(mcfg["resolution"])
        self.origin_x = float(mcfg["origin"][0])
        self.origin_y = float(mcfg["origin"][1])

        free_idx = np.argwhere(~self.occ)
        self.free_cells = [(int(r), int(c)) for r, c in free_idx]
        if len(self.free_cells) == 0:
            raise RuntimeError("No free cell found in map.")

    def _world_to_grid(self, x, y):
        c = int((x - self.origin_x) / self.resolution)
        r = int(self.height - 1 - (y - self.origin_y) / self.resolution)
        return r, c

    def _grid_to_world(self, r, c):
        x = self.origin_x + (c + 0.5) * self.resolution
        y = self.origin_y + (self.height - 1 - r + 0.5) * self.resolution
        return x, y

    def _in_map(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def _is_free(self, x, y):
        r, c = self._world_to_grid(x, y)
        if not self._in_map(r, c):
            return False
        return not self.occ[r, c]

    def _sample_free_pose(self):
        r, c = self.rng.choice(self.free_cells)
        x, y = self._grid_to_world(r, c)
        yaw = self.rng.uniform(-math.pi, math.pi)
        return x, y, yaw

    def _sample_start_goal(self):
        for _ in range(2000):
            sx, sy, syaw = self._sample_free_pose()
            gx, gy, _ = self._sample_free_pose()
            d = math.hypot(gx - sx, gy - sy)
            if self.min_start_goal_dist <= d <= self.max_start_goal_dist:
                return sx, sy, syaw, gx, gy
        raise RuntimeError("Failed to sample start/goal pair.")

    def _raycast(self, angle):
        d = 0.0
        while d <= self.lidar_max_range:
            x = self.x + d * math.cos(angle)
            y = self.y + d * math.sin(angle)
            if not self._is_free(x, y):
                return d
            d += self.ray_step
        return self.lidar_max_range

    def _scan_three_beams(self):
        front = self._raycast(self.yaw)
        left = self._raycast(self.yaw + math.radians(self.lidar_front_deg))
        right = self._raycast(self.yaw - math.radians(self.lidar_front_deg))
        return front, left, right

    def _build_obs(self):
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist_goal = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        yaw_err = wrap_angle(heading - self.yaw)
        front, left, right = self._scan_three_beams()
        obs = np.array([dist_goal, yaw_err, front, left, right, self.v, self.w], dtype=np.float32)
        return obs

    def _stack_obs(self, obs):
        if len(self.obs_history) == 0:
            for _ in range(self.frame_stack):
                self.obs_history.append(obs.copy())
        else:
            self.obs_history.append(obs.copy())
        return np.concatenate(list(self.obs_history), axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.prev_action[:] = 0.0
        self.obs_history.clear()
        self.ep_r_progress = 0.0
        self.ep_r_step = 0.0
        self.ep_r_align = 0.0
        self.ep_r_speed = 0.0
        self.ep_r_safe = 0.0
        self.ep_r_spin = 0.0
        self.ep_r_stall = 0.0
        self.ep_r_collision = 0.0
        self.ep_r_goal = 0.0
        sx, sy, syaw, gx, gy = self._sample_start_goal()
        self.x, self.y, self.yaw = sx, sy, syaw
        self.v, self.w = 0.0, 0.0
        self.goal = np.array([gx, gy], dtype=np.float32)

        obs = self._build_obs()
        self.prev_dist_goal = float(obs[0])
        return self._stack_obs(obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        v_cmd = float(np.clip(action[0], -self.max_lin, self.max_lin))
        w_cmd = float(np.clip(action[1], -self.max_ang, self.max_ang))

        self.v = v_cmd
        self.w = w_cmd
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw = wrap_angle(self.yaw + self.w * self.dt)
        self.step_count += 1

        obs = self._build_obs()
        dist_goal = float(obs[0])
        yaw_err = float(obs[1])
        front = float(obs[2])
        left = float(obs[3])
        right = float(obs[4])

        collision = (front <= self.collision_radius) or (not self._is_free(self.x, self.y))
        reached = dist_goal <= self.goal_tolerance
        timeout = self.step_count >= self.max_steps

        progress = self.prev_dist_goal - dist_goal
        min_laser_dist = max(min(front, left, right), 1e-3)

        # 1) 主驱动：前进进度（最大项）
        r_progress = self.progress_weight * progress
        # 2) 朝向 + 前进耦合（不前进不奖励，方向错则扣分）
        r_align = self.heading_weight * v_cmd * math.cos(yaw_err)
        # 3) 速度激励（防止磨蹭）
        r_speed = self.speed_weight * v_cmd
        # 4) 停滞惩罚（专门抑制慢磨策略）
        r_stall = self.stall_penalty if progress < self.stall_progress_eps else 0.0
        # 5) 避障惩罚：离障碍越近惩罚越大
        r_safe = -self.safe_weight * (1.0 / min_laser_dist)
        # 6) 转向约束：原地转强罚，前进中转轻罚
        spin_scale = 1.0 if v_cmd < self.spin_stop_v_thresh else 0.2
        r_spin = -self.turn_weight * abs(w_cmd) * spin_scale
        # 7) 时间惩罚（防拖时间）
        r_step = self.step_penalty
        # 8) 终止项：碰撞惩罚 + 到达奖励
        r_collision = self.collision_penalty if collision else 0.0
        r_goal = self.goal_reward if reached else 0.0

        # 总奖励（用于 PPO 优化）
        reward = (
            r_progress
            + r_align
            + r_speed
            + r_safe
            + r_spin
            + r_stall
            + r_step
            + r_collision
            + r_goal
        )

        self.ep_r_progress += r_progress
        self.ep_r_align += r_align
        self.ep_r_speed += r_speed
        self.ep_r_safe += r_safe
        self.ep_r_spin += r_spin
        self.ep_r_stall += r_stall
        self.ep_r_step += r_step
        self.ep_r_collision += r_collision
        self.ep_r_goal += r_goal

        self.prev_action = np.array([v_cmd, w_cmd], dtype=np.float32)
        self.prev_dist_goal = dist_goal

        terminated = collision or reached
        truncated = timeout
        info = {
            "collision": collision,
            "reached": reached,
            "timeout": timeout,
            "r_progress": float(r_progress),
            "r_align": float(r_align),
            "r_speed": float(r_speed),
            "r_safe": float(r_safe),
            "r_spin": float(r_spin),
            "r_stall": float(r_stall),
            "r_step": float(r_step),
            "r_collision": float(r_collision),
            "r_goal": float(r_goal),
            # Backward-compatible aliases for old logger keys.
            "r_heading": float(r_align),
            "r_turn": float(r_spin),
            "r_total": float(reward),
        }
        if terminated or truncated:
            info["ep_r_progress"] = float(self.ep_r_progress)
            info["ep_r_align"] = float(self.ep_r_align)
            info["ep_r_speed"] = float(self.ep_r_speed)
            info["ep_r_safe"] = float(self.ep_r_safe)
            info["ep_r_spin"] = float(self.ep_r_spin)
            info["ep_r_stall"] = float(self.ep_r_stall)
            info["ep_r_step"] = float(self.ep_r_step)
            info["ep_r_collision"] = float(self.ep_r_collision)
            info["ep_r_goal"] = float(self.ep_r_goal)
            info["ep_r_heading"] = float(self.ep_r_align)
            info["ep_r_turn"] = float(self.ep_r_spin)
            info["ep_r_total"] = float(
                self.ep_r_progress
                + self.ep_r_align
                + self.ep_r_speed
                + self.ep_r_safe
                + self.ep_r_spin
                + self.ep_r_stall
                + self.ep_r_step
                + self.ep_r_collision
                + self.ep_r_goal
            )
        return self._stack_obs(obs), reward, terminated, truncated, info
