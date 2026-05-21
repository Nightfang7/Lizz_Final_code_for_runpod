#!/bin/bash
# 啟動監控式訓練的腳本

DATASET=${1:-split4}
SHOTS=${2:-5}
TARGET=${3:-0.9}
RESULTS_DIR=${4:-results}
CHECK_INTERVAL=${5:-30}
WAYS=${6:-5}

echo "Starting monitored training:"
echo "  Dataset: $DATASET"
echo "  Shots: $SHOTS"
echo "  Ways: $WAYS"
echo "  Target accuracy: $TARGET"
echo "  Results directory: $RESULTS_DIR"
echo "  Check interval: $CHECK_INTERVAL seconds"
echo ""

# 創建結果目錄
mkdir -p "$RESULTS_DIR"

# 啟動監控腳本（後台運行）
echo "Starting accuracy monitor..."
python monitor_accuracy.py "$RESULTS_DIR" --target "$TARGET" --dataset "$DATASET" --shots "$SHOTS" --ways "$WAYS" --check_interval "$CHECK_INTERVAL" &
MONITOR_PID=$!

# 等待一下讓監控腳本啟動
sleep 2

# 啟動訓練腳本
echo "Starting training..."
python train_with_stop.py "$DATASET" "$SHOTS" --results_dir "$RESULTS_DIR"

# 等待監控腳本結束
wait $MONITOR_PID

echo "Training session completed!"
