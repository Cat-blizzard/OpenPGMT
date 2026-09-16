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
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=isaacgym_common.sh
source "$SCRIPT_DIR/isaacgym_common.sh"

ENV_NAME=pgmt
ISAACGYM_TAR=""
PYTHON_VER=""
SKIP_SMOKE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --isaacgym|--python)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[!] $1 需要取值"; exit 1
      fi
      if [[ "$1" == "--isaacgym" ]]; then ISAACGYM_TAR="$2"; else PYTHON_VER="$2"; fi
      shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$ISAACGYM_TAR" ]]; then
  echo "[!] 需要 --isaacgym /path/IsaacGym_Preview_4_Package.tar.gz"
  exit 1
fi
if [[ ! -f "$ISAACGYM_TAR" ]]; then
  echo "[!] Isaac Gym 包不存在: $ISAACGYM_TAR"; exit 1
fi
# Resolve relative paths before changing to the repository directory.
ISAACGYM_TAR="$(cd "$(dirname "$ISAACGYM_TAR")" && pwd)/$(basename "$ISAACGYM_TAR")"
PYTHON_VER=$(isaacgym_select_python "$ISAACGYM_TAR" "$PYTHON_VER")
cd "$SCRIPT_DIR/.."

echo "===== 0. 硬件/驱动检查 ====="
if ! command -v nvidia-smi >/dev/null; then
  echo "[!] 未检测到 nvidia-smi，请在 GPU 服务器上运行本脚本"; exit 1
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo "===== 1. Python 版本（已核对包内绑定） ====="
echo "[i] 将创建 conda 环境 python=$PYTHON_VER"

if conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
  echo "[i] conda 环境 ${ENV_NAME} 已存在，复用"
else
  conda create -n ${ENV_NAME} python=${PYTHON_VER} -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
ACTIVE_PYTHON=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [[ "$ACTIVE_PYTHON" != "$PYTHON_VER" ]]; then
  echo "[!] 已有环境 $ENV_NAME 使用 Python $ACTIVE_PYTHON，但包要求 $PYTHON_VER。请先修正环境。"
  exit 1
fi

echo "===== 2. torch 2.1.2 (cu121) ====="
# 官方 PP4 绑定按旧 torch ABI 编译，必须用 2.1 时代版本；cu121 驱动 580 兼容
if python -c 'import torch; assert torch.__version__.split("+")[0] == "2.1.2" and torch.version.cuda == "12.1"' 2>/dev/null; then
  echo "[i] torch 2.1.2 cu121 已安装，跳过"
else
  python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print(f'[i] torch {torch.__version__}, cuda {torch.version.cuda}')"

echo "===== 3. 其余依赖 ====="
python -m pip install -r setup/requirements.txt "torch==2.1.2"
python -c 'import torch; assert torch.__version__.split("+")[0] == "2.1.2" and torch.version.cuda == "12.1", "依赖安装改变了 torch/CUDA 版本"'

echo "===== 4. Isaac Gym ====="
if python -c "import isaacgym" 2>/dev/null; then
  echo "[i] isaacgym 已可用，跳过安装"
else
  TMPD=$(mktemp -d)
  trap 'rm -rf "$TMPD"' EXIT
  tar --force-local -xf "$ISAACGYM_TAR" -C "$TMPD"
  # 注意：-e（editable）会在 site-packages 留一个指向上面的临时目录的
  # egg-link，/tmp 被清理后 import 失效。因此解压到持久位置再 -e 安装。
  DEST="${ISAACGYM_DIR:-$HOME/isaacgym}"
  if [[ -e "$DEST" ]]; then
    if [[ ! -f "$DEST/python/setup.py" ]]; then
      echo "[!] 已有目录不是 Isaac Gym 源码目录: $DEST。请通过 ISAACGYM_DIR 指定新位置。"
      exit 1
    fi
    echo "[i] 复用现有源码目录: $DEST"
  else
    mkdir -p "$(dirname "$DEST")"
    cp -r "$TMPD/isaacgym" "$DEST"
    echo "[i] 包已解压到持久路径: $DEST（避免 /tmp 清理后 egg-link 失效）"
  fi
  python -m pip install -e "$DEST/python"
fi

echo "===== 5. 冒烟测试 ====="
if [[ "$SKIP_SMOKE" == "0" ]]; then
  python setup/smoke_test.py
else
  echo "[i] 已跳过冒烟测试"
fi

echo "===== 完成 ====="
echo "conda activate ${ENV_NAME} && python setup/smoke_test.py   # 随时重跑冒烟"
echo "跑本机单测: pip install pytest matplotlib && python -m pytest tests -q"
echo "多卡并行: 每个训练 run 一个进程, CUDA_VISIBLE_DEVICES=k python -m pgmt.train.train_stage1（M2 交付后可用）"
