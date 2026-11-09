#!/usr/bin/env bash
# 6P + 2D：P 默认 DP3 × TP2 + EP6，D 默认 DP2 × TP1 + EP2。
# ENABLE_EP 同时作用于 P/D；其余配置复用 capture_pd.sh。DRY_RUN=1 可预览。
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

# ==================== 设备 / 并行 ====================
export P_GPUS="${P_GPUS:-0,1,2,3,4,5}" P_TP="${P_TP:-2}" P_DP="${P_DP:-3}"
export D_GPUS="${D_GPUS:-6,7}" D_TP="${D_TP:-1}" D_DP="${D_DP:-2}"
export ENABLE_EP="${ENABLE_EP:-1}"

# 负载、token budget、NIXL、端口、trace、模型和输出目录沿用原入口，
# 均可通过环境变量覆盖。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CAPTURE_ENTRYPOINT="$script_dir/capture_pd_6p2d.sh"
exec bash "$script_dir/capture_pd.sh"
