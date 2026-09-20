#!/usr/bin/env bash
# ============================================================
# PGMT 服务器安装 —— 路径 A（兜底）: Isaac Lab
#
# 适用: setup/check_isaacgym.sh 判定 PP4 不可行（或 PP4 冒烟失败且无法降驱动）。
# Isaac Lab 官方支持 Ada/Blackwell + 新驱动（580）+ cu128。
#
# 安装内容:
#   conda env "pgmt-lab" (python 3.11)
#   isaaclab[isaacsim,all]==2.3.2.post1  (Isaac Sim 5.1.0)
#   torch 2.7.0 cu128（按 Isaac Lab 官方 pip 安装文档）
#   冒烟测试: setup/smoke_test_isaaclab.py
#
# 说明: Isaac Lab 对 Blackwell 的官方支持自 2.1.1 起（torch 2.7+cu128）。
#   若后续想用 rsl-rl 2.x API 写训练代码，可改装 Isaac Lab 2.2.1
#   （对应 rsl-rl-lib==2.3.3）；默认装最新稳定版。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME=pgmt-lab
ISAACLAB_VER="${ISAACLAB_VER:-2.3.2.post1}"
WARP_VER="${WARP_VER:-1.12.1}"
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "===== 0. 硬件/驱动检查 ====="
if ! command -v nvidia-smi >/dev/null; then
  echo "[!] 未检测到 nvidia-smi，请在 GPU 服务器上运行本脚本"; exit 1
fi
nvidia-smi

echo "===== 1. conda 环境 (python 3.11) ====="
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "[i] conda 环境 ${ENV_NAME} 已存在，复用"
else
  conda create -n ${ENV_NAME} python=3.11 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "===== 2. Isaac Lab ====="
if python -c "import isaaclab" 2>/dev/null; then
  echo "[i] isaaclab 已安装，跳过"
else
  pip install "isaaclab[isaacsim,all]==${ISAACLAB_VER}" --extra-index-url https://pypi.nvidia.com
fi

# Isaac Sim 5.1's Python utilities still annotate with ``warp.types.array``;
# newer pip Warp releases removed that public alias.  Keep the Isaac Sim 5.1
# environment reproducible instead of allowing pip to resolve an incompatible
# latest Warp release.  Override WARP_VER only when moving the simulator stack.
pip install --force-reinstall --no-deps "warp-lang==${WARP_VER}"
python - <<'PY'
import warp
assert hasattr(warp.types, "array"), "warp.types.array is required by Isaac Sim 5.1"
print(f"[i] warp {getattr(warp, '__version__', 'unknown')} (Isaac Sim 5.1 compatible API)")
PY

echo "===== 3. torch (cu128) ====="
# 注意顺序: 先装 isaaclab 再装 torch，按 Isaac Lab 官方文档
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(f'[i] torch {torch.__version__}, cuda {torch.version.cuda}')"

echo "===== 4. 冒烟测试 ====="
if [[ "$SKIP_SMOKE" == "0" ]]; then
  python setup/smoke_test_isaaclab.py
else
  echo "[i] 已跳过冒烟测试"
fi

echo "===== 完成 ====="
echo "conda activate ${ENV_NAME} && python setup/smoke_test_isaaclab.py   # 随时重跑冒烟"
echo "下一步: M2 阶段按 Isaac Lab API 编写环境（manager-based），训练配置用 isaaclab_rl。"
