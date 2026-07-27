"""Stable public names for TriSched models and reports.

Snake-case identifiers such as ``task_gnn`` remain internal compatibility
keys for frozen checkpoints and historical evidence packages. New user-facing
text must use the display names defined here.
"""

PROJECT_DISPLAY_NAME = "TriSched"
PROJECT_DESCRIPTION = "面向云—边—端异构计算资源管理调度方法"

TRISCHED_GNN_PPO_DISPLAY_NAME = "TriSched-GNN-PPO"
TRISCHED_GNN_PPO_CLI_COMMAND = "train-trisched-gnn-ppo"

MASKED_MLP_DISPLAY_NAME = "Masked MLP"
MASKED_MLP_PAPER_ROLE = "TriSched-MLP-PPO消融基线"

# Frozen evidence and checkpoint compatibility identifiers.
LEGACY_TASK_GNN_ID = "task_gnn"
LEGACY_TASK_GNN_ARCHITECTURE = "task_gnn_v1"
LEGACY_TASK_GNN_CLI_COMMAND = "train-task-gnn"
