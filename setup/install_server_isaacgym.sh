#!/usr/bin/env bash
# ============================================================
# PGMT 服务器安装 —— 路径 B: 官方 Isaac Gym Preview 4 + RTX 5880 Ada
#
# 前置:
#   1) 从 NVIDIA developer 网站（免费注册）下载 IsaacGym_Preview_4_Package.tar.gz
#   2) bash setup/check_isaacgym.sh <tar>   # 确认绑定版本与 sm_89 内核
#
# 用法:
#   bash setup/install_server_isaacgym.sh --isaacgym /path/IsaacGym_Preview_4_Package.tar.gz
#   bash setup/install_server_isaacgym.sh --isaacgym <tar> --skip-smoke
#
# 组合（Ada 社区黄金配置）:
#   python 3.9（官方绑定 gym_39.so；若无则 3.8）
#   torch 2.1.2 cu121（官方绑定按旧 torch ABI 编译，勿用 2.7+cu128）
#   rsl-rl-lib 2.1.2（legged_gym 标准搭配）
#
# 驱动注意: 本机驱动 580 超出社区验证范围（黄金线 525/535，但 headless
#   训练在 555+ 有先例）。冒烟测试一跑便知；若 create_sim 崩溃再考虑
#   降驱动（5880 Ada 最低约 535/545）或转 Isaac Lab（install_server.sh）。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME=pgmt
ISAACGYM_TAR=""
PYTHON_VER=""
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --isaacgym) ISAACGYM_TAR="$2"; shift 2 ;;
    --python) PYTHON_VER="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$ISAACGYM_TAR" ]]; then
  echo "[!] 需要 --isaacgym /path/IsaacGym_Preview_4_Package.tar.gz"
  exit 1
fi

echo "===== 0. 硬件/驱动检查 ====="
if ! command -v nvidia-smi >/dev/null; then
  echo "[!] 未检测到 nvidia-smi，请在 GPU 服务器上运行本脚本"; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo "===== 1. 自动确定 Python 版本（按官方绑定） ====="
if [[ -z "$PYTHON_VER" ]]; then
  MAXPY=$(tar -tzf "$ISAACGYM_TAR" | grep -oE "gym_3[0-9]\.so" | grep -oE "3[0-9]" | sort -n | tail -1)
  PYTHON_VER="3.${MAXPY:-8}"
  echo "[i] 官方绑定最高支持 Python $PYTHON_VER"
fi

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "[i] conda 环境 ${ENV_NAME} 已存在，复用"
else
  conda create -n ${ENV_NAME} python=${PYTHON_VER} -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "===== 2. torch 2.1.2 (cu121) ====="
# 官方 PP4 绑定按旧 torch ABI 编译，必须用 2.1 时代版本；cu121 驱动 580 兼容
if python -c "import torch" 2>/dev/null; then
  echo "[i] torch 已安装，跳过"
else
  pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print(f'[i] torch {torch.__version__}, cuda {torch.version.cuda}')"

echo "===== 3. 其余依赖 ====="
pip install -r setup/requirements.txt

echo "===== 4. Isaac Gym ====="
if python -c "import isaacgym" 2>/dev/null; then
  echo "[i] isaacgym 已可用，跳过安装"
else
  TMPD=$(mktemp -d)
  tar -xzf "$ISAACGYM_TAR" -C "$TMPD"
  pip install -e "$TMPD/isaacgym/python" || pip install "$TMPD/isaacgym/python"
fi

echo "===== 5. 冒烟测试 ====="
if [[ "$SKIP_SMOKE" == "0" ]]; then
  python setup/smoke_test.py
else
  echo "[i] 已跳过冒烟测试"
fi

echo "===== 完成 ====="
echo "conda activate ${ENV_NAME} && python setup/smoke_test.py   # 随时重跑冒烟"
echo "多卡并行: 每个训练 run 一个进程, CUDA_VISIBLE_DEVICES=k python pgmt/train/train_stage1.py"
