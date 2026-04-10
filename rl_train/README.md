# RL Train (2D Gym)

这个目录用于训练 2D 局部导航策略（不依赖 ROS 运行时），训练完成后导出模型给
`src/me5413_world/scripts/policy_infer.py` 使用。

## 目录

- `configs/nav_rl.yaml`: 训练与环境参数
- `envs/nav2d_env.py`: Gymnasium 2D 导航环境
- `train/train_ppo.py`: Stable-Baselines3 PPO 训练入口
- `export/export_torchscript.py`: 导出 TorchScript (`.pt`) 模型
- `checkpoints/`: 训练输出（建议加入 `.gitignore`）

## 1) 安装依赖

```bash
cd /home/yun/me5413/ME5413_Final_Project/rl_train
# 推荐：使用你现有 conda 环境
# conda activate slosh

# 或者单独 venv
# python3 -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2) 开始训练

```bash
python train/train_ppo.py --config configs/nav_rl.yaml
```

### 可选：使用 Weights & Biases 看训练曲线

先登录：

```bash
wandb login
```

有两种开启方式：

1) 在 `configs/nav_rl.yaml` 中设置：

```yaml
train:
  wandb:
    enabled: true
    project: me5413-nav2d
```

2) 命令行临时开启：

```bash
python train/train_ppo.py --config configs/nav_rl.yaml --wandb
```

W&B 会自动同步 TensorBoard 指标（`rollout/*`, `train/*`, `time/*`）。

训练结果默认保存到：

- `checkpoints/ppo_nav2d_final.zip`
- `checkpoints/obs_spec.json`

## 3) 导出 TorchScript

```bash
python export/export_torchscript.py \
  --ckpt checkpoints/ppo_nav2d_final.zip \
  --obs-spec checkpoints/obs_spec.json \
  --out checkpoints/policy.pt
```

## 4) 部署到 ROS 工作空间

把导出的 `policy.pt` 放到主仓库，例如：

```bash
cp checkpoints/policy.pt /home/yun/me5413/ME5413_Final_Project/models/policy.pt
```

然后在 `src/me5413_world/launch/navigation_rl.launch` 里设置：

- `policy_backend = torchscript`
- `policy_model_path = /home/yun/me5413/ME5413_Final_Project/models/policy.pt`

## 观测顺序（必须一致）

这里与 `policy_infer.py` 对齐，顺序固定为：

1. `dist_goal`
2. `yaw_err`
3. `front`
4. `left`
5. `right`
6. `v_curr`
7. `w_curr`

动作顺序为：

1. `linear_x` (m/s)
2. `angular_z` (rad/s)

## 时间序列观测（帧堆叠）

通过 `configs/nav_rl.yaml` 中的 `obs.frame_stack` 配置（默认 4），
训练和部署都会把最近 K 帧观测拼接后输入策略网络。  
如果你修改了 `frame_stack`，需要重新训练并重新导出模型。
