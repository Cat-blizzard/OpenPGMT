#!/usr/bin/env bash
# ============================================================
# 决策树第 1 步: 检查官方 Isaac Gym Preview 4 包在目标机器上的可行性
#
# 检查两件事:
#   1) Python 绑定版本: 列出包内 gym_3X.so，确定官方绑定支持的最高 Python 版本
#   2) PhysX GPU 内核: libPhysXGpu_64.so 是否含目标 GPU 架构的 cubin 或 PTX
#      （目标架构默认从 nvidia-smi 自动检测，如 5880 Ada -> sm_89；
#        也可 --arch sm_120 显式指定，如检查 5090）
#
# 用法:
#   bash setup/check_isaacgym.sh /path/IsaacGym_Preview_4_Package.tar.gz [--arch sm_89]
#
# 结果解读:
#   - 绑定只有 gym_38.so  -> 服务器 Python 用 3.8
#   - 绑定有 gym_39.so    -> 服务器 Python 可用 3.9（torch 2.1.2/numpy 1.26 兼容）
#   - 静态检查只提供候选兼容性，最终以目标服务器冒烟为准
# ============================================================
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=isaacgym_common.sh
source "$SCRIPT_DIR/isaacgym_common.sh"

if [[ $# -lt 1 ]]; then
  echo "用法: bash setup/check_isaacgym.sh /path/IsaacGym_Preview_4_Package.tar.gz [--arch sm_89]"
  exit 1
fi
TAR="$1"
ARCH=""
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      if [[ $# -lt 2 || ! "$2" =~ ^sm_[0-9]+$ ]]; then
        echo "[!] --arch 需要取值，例如 --arch sm_89"; exit 1
      fi
      ARCH="$2"; shift 2 ;;
    *) echo "[!] 未知参数: $1"; exit 1 ;;
  esac
done
BINDINGS=$(isaacgym_python_versions "$TAR")
if [[ -z "$BINDINGS" ]]; then
  echo "[!] 包内没有 Python 绑定"; exit 1
fi

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
echo "[i] 解压 $TAR ..."
tar --force-local -xf "$TAR" -C "$TMPD"

echo ""
echo "===== 1. Python 绑定版本 ====="
echo "$BINDINGS" | sed 's/^/  Python /'
MAXPY=$(printf '%s\n' "$BINDINGS" | tail -n 1)
echo "[结论] 官方绑定支持的最高 Python: $MAXPY"

echo ""
echo "===== 2. PhysX GPU 内核架构 ====="
if [[ -z "$ARCH" ]]; then
  if command -v nvidia-smi >/dev/null; then
    # 从当前 GPU 计算能力推导（sm_89 -> 89）
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p' | tr -d '.[:space:]')
    ARCH="sm_${CC}"
    echo "[i] 自动检测目标架构: $ARCH"
  else
    ARCH="sm_89"
    echo "[i] 无 nvidia-smi，默认按 Ada 检查: $ARCH"
  fi
fi
ARCH_NUM=$(echo "$ARCH" | grep -oE "[0-9]+")

if ! command -v cuobjdump >/dev/null; then
  echo "[!] 未找到 cuobjdump（需 CUDA toolkit），跳过内核检查"
  echo "    备选: 直接跑 smoke_test.py 验证（import 失败/no kernel image 即不可行）"
else
  SO=$(find "$TMPD" -name "libPhysXGpu_64.so" | sed -n '1p')
  if [[ -z "$SO" ]]; then
    echo "[!] 未在包内找到 libPhysXGpu_64.so"
  else
    echo "[i] 检查: $SO"
    echo "--- cubin 架构 (--list-elf) ---"
    ELF_LIST=$(cuobjdump --list-elf "$SO" 2>/dev/null || true)
    PTX_LIST=$(cuobjdump --list-ptx "$SO" 2>/dev/null || true)
    printf '%s\n' "$ELF_LIST" | grep -oE "sm_[0-9]+" | sort -u || echo "(无)"
    echo "--- PTX 架构 (--list-ptx) ---"
    printf '%s\n' "$PTX_LIST" | grep -oE "(compute|sm)_[0-9]+" | sort -u || echo "(无)"

    echo ""
    if printf '%s\n' "$ELF_LIST" | grep -oE 'sm_[0-9]+' | grep -Fx "$ARCH" >/dev/null; then
      echo "[结论] 含 $ARCH cubin；仍需服务器冒烟确认"
    else
      # PTX 向前兼容: 低版本 compute_X 的 PTX 可 JIT 到更高架构。
      # 必须按数值比较，不能字典序（compute_100 < compute_90，三字节架构号会取错）
      MINPTX=$(printf '%s\n' "$PTX_LIST" | grep -oE "(compute|sm)_[0-9]+" \
        | grep -oE "[0-9]+" | sort -n | sed -n '1p' || true)
      if [[ -n "$MINPTX" ]]; then
        if [[ "$MINPTX" -le "$ARCH_NUM" ]]; then
          echo "[结论] 含不高于目标的 PTX compute_$MINPTX，可能支持 JIT；仍需服务器冒烟确认"
        else
          echo "[结论] 未发现目标架构 cubin 或可用的前向 PTX；请运行冒烟确认，失败时考虑 Isaac Lab"
        fi
      else
        echo "[结论] 未发现目标架构 cubin 或 PTX；请运行冒烟确认，失败时考虑 Isaac Lab"
      fi
    fi
  fi
fi

echo ""
echo "===== 下一步 ====="
echo "bash setup/install_server_isaacgym.sh --isaacgym $TAR"
