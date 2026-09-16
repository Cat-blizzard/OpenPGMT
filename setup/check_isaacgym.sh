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
#   - 无目标架构内核/PTX  -> 该 GPU 跑不了 PP4，转 Isaac Lab
# ============================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: bash setup/check_isaacgym.sh /path/IsaacGym_Preview_4_Package.tar.gz [--arch sm_89]"
  exit 1
fi
TAR="$1"
ARCH=""
if [[ $# -ge 2 && "$2" == "--arch" ]]; then
  if [[ $# -lt 3 ]]; then
    echo "[!] --arch 需要取值，例如 --arch sm_89"
    exit 1
  fi
  ARCH="$3"
fi

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
echo "[i] 解压 $TAR ..."
tar -xzf "$TAR" -C "$TMPD"

echo ""
echo "===== 1. Python 绑定版本 ====="
BINDINGS=$(find "$TMPD" -name "gym_3[0-9].so" -printf "%f\n" 2>/dev/null | sort || \
           find "$TMPD" -name "gym_3[0-9].so" | xargs -r -n1 basename | sort)
if [[ -z "$BINDINGS" ]]; then
  echo "[!] 未找到 gym_3X.so 绑定，包结构可能异常"
else
  echo "$BINDINGS" | sed 's/^/  /'
  # gym_39.so 中 "39" 的 3 已是主版本号，不可再拼 "3." 前缀（旧写法会打印 3.39）
  MAXPY=$(echo "$BINDINGS" | grep -oE "[0-9]+" | sort -n | tail -1)
  echo "[结论] 官方绑定支持的最高 Python: 3.$MAXPY"
fi

echo ""
echo "===== 2. PhysX GPU 内核架构 ====="
if [[ -z "$ARCH" ]]; then
  if command -v nvidia-smi >/dev/null; then
    # 从当前 GPU 计算能力推导（sm_89 -> 89）
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
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
  SO=$(find "$TMPD" -name "libPhysXGpu_64.so" | head -1)
  if [[ -z "$SO" ]]; then
    echo "[!] 未在包内找到 libPhysXGpu_64.so"
  else
    echo "[i] 检查: $SO"
    echo "--- cubin 架构 (--list-elf) ---"
    cuobjdump --list-elf "$SO" | grep -oE "sm_[0-9]+" | sort -u || echo "(无)"
    echo "--- PTX 架构 (--list-ptx) ---"
    cuobjdump --list-ptx "$SO" | grep -oE "compute_[0-9]+" | sort -u || echo "(无)"

    echo ""
    if cuobjdump --list-elf "$SO" | grep -qE "sm_${ARCH_NUM}"; then
      echo "[结论] 含 $ARCH cubin —— PP4 可直接运行"
    elif cuobjdump --list-ptx "$SO" | grep -qE "compute_${ARCH_NUM}"; then
      echo "[结论] 含 compute_${ARCH_NUM} PTX（可 JIT）—— PP4 可运行"
    else
      # PTX 向前兼容: 低版本 compute_X 的 PTX 可 JIT 到更高架构。
      # 必须按数值比较，不能字典序（compute_100 < compute_90，三字节架构号会取错）
      MAXPTX=$(cuobjdump --list-ptx "$SO" | grep -oE "compute_[0-9]+" \
        | grep -oE "[0-9]+" | sort -n | tail -1)
      if [[ -n "$MAXPTX" ]]; then
        if [[ "$MAXPTX" -le "$ARCH_NUM" ]]; then
          echo "[结论] 最高 PTX 为 compute_$MAXPTX，可 JIT 至 $ARCH（PTX 向前兼容）—— PP4 可运行"
        else
          echo "[结论] 无 $ARCH 内核且 PTX 不向前兼容（最高 compute_$MAXPTX）—— PP4 在本机 GPU 上无解，转 Isaac Lab"
        fi
      else
        echo "[结论] 无 $ARCH 内核且无任何 PTX —— PP4 在本机 GPU 上无解，转 Isaac Lab"
      fi
    fi
  fi
fi

echo ""
echo "===== 下一步 ====="
echo "bash setup/install_server_isaacgym.sh --isaacgym $TAR"
