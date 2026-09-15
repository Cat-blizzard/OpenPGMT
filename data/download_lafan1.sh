#!/usr/bin/env bash
# 下载 LAFAN1（Ubisoft LaForge Animation Dataset，Mixamo BVH，~144MB）
# 到 data/raw/ 并解压为 data/raw/lafan1/*.bvh。
#
# 数据托管在 GitHub LFS（ubisoft/ubisoft-laforge-animation-dataset），
# 直接 raw 下载得到的是 LFS 指针，须用 media 地址。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/raw"
ZIP="$RAW_DIR/lafan1.zip"
URL="https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip"

mkdir -p "$RAW_DIR"

if [ -f "$ZIP" ] && [ "$(stat -c%s "$ZIP" 2>/dev/null || stat -f%z "$ZIP")" -gt 100000000 ]; then
    echo "[skip] $ZIP 已存在"
else
    echo "[下载] $URL"
    curl -L --retry 3 -o "$ZIP" "$URL"
fi

echo "[解压] $ZIP -> $RAW_DIR/lafan1/"
mkdir -p "$RAW_DIR/lafan1"
unzip -o -q "$ZIP" -d "$RAW_DIR/lafan1"
echo "[完成] $(ls "$RAW_DIR/lafan1"/*.bvh | wc -l) 个 BVH 文件"
echo "下一步: python -m data.retarget_lafan1 --bvh-dir data/raw/lafan1"
