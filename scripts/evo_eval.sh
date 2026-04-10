#!/bin/bash
# EVO 轨迹评估脚本：用 EVO 比较 NDT 估计轨迹与 Gazebo 真值
# 用法：
#   1) 仅评估（已有 bag）：  ./scripts/evo_eval.sh <bag文件>
#   2) 查看帮助：            ./scripts/evo_eval.sh -h

set -e
BAG="${1:-}"
GT_TOPIC="/gazebo/ground_truth/state"
EST_TOPIC="/ndt_pose"

usage() {
  echo "用法: $0 <bag文件>"
  echo "  用 EVO 计算 ATE/RPE，并可选绘图。"
  echo "  真值话题: $GT_TOPIC  估计话题: $EST_TOPIC"
  exit 0
}

if [[ "$BAG" == "-h" || "$BAG" == "--help" || -z "$BAG" ]]; then
  usage
fi

if [[ ! -f "$BAG" ]]; then
  echo "错误: 找不到 bag 文件: $BAG"
  exit 1
fi

echo "=== EVO 评估: $BAG ==="
echo "真值: $GT_TOPIC  估计: $EST_TOPIC"
echo ""

echo "--- ATE (绝对轨迹误差, 平移) ---"
evo_ape bag "$BAG" "$GT_TOPIC" "$EST_TOPIC" --pose_relation trans_part -va --align

echo ""
echo "--- RPE (相对位姿误差, 平移) ---"
evo_rpe bag "$BAG" "$GT_TOPIC" "$EST_TOPIC" --pose_relation trans_part -va --align

echo ""
echo "--- 轨迹对比图 (若未自动弹出可忽略) ---"
evo_traj bag "$BAG" "$GT_TOPIC" "$EST_TOPIC" -p

echo "评估完成。"
