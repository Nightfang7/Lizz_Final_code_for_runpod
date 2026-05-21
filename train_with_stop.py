#!/usr/bin/env python3
"""
帶停止信號的訓練腳本
持續運行單個實驗，直到收到停止信號

使用方法:
python train_with_stop.py split4 5 --cleanup
"""

import argparse
import os
import sys
import time
from pathlib import Path

def check_stop_signal(results_dir):
    """檢查是否有停止信號"""
    stop_file = os.path.join(results_dir, "STOP_TRAINING.txt")
    return os.path.exists(stop_file)

def main():
    parser = argparse.ArgumentParser(description='Train with stop signal')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('shots', type=int, help='Number of shots')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup intermediate artifacts')
    parser.add_argument('--results_dir', default='results', help='Results directory')
    parser.add_argument('--sleep_seconds', type=float, default=5, help='Sleep between experiments')
    parser.add_argument('--keep_folders', default='11_validation_results_in_train_csv,15_test_results_after_train_csv,metrics_output,logs', help='Folders to keep during cleanup')
    
    args = parser.parse_args()
    
    print(f"Starting continuous training for {args.dataset} {args.shots}shot")
    print(f"Results directory: {args.results_dir}")
    print(f"Cleanup: {args.cleanup}")
    print("Press Ctrl+C to stop manually")
    print("=" * 60)
    
    experiment_count = 0
    
    try:
        while True:
            # 檢查停止信號
            if check_stop_signal(args.results_dir):
                print(f"\nStop signal detected! Stopping training.")
                break
            
            experiment_count += 1
            print(f"\n[Experiment {experiment_count}] Starting...")
            
            # 構建命令
            cmd_parts = [
                "python", "run_experiments.py", "single",
                args.dataset, str(args.shots), "unix"
            ]
            
            if args.cleanup:
                cmd_parts.append("--cleanup")
                cmd_parts.append("--keep_folders")
                cmd_parts.append(f'"{args.keep_folders}"')
            
            cmd = " ".join(cmd_parts)
            print(f"Running: {cmd}")
            
            # 執行訓練
            import subprocess
            result = subprocess.run(cmd, shell=True, capture_output=False)
            
            if result.returncode == 0:
                print(f"[Experiment {experiment_count}] Completed successfully")
            else:
                print(f"[Experiment {experiment_count}] Failed with return code {result.returncode}")
            
            # 再次檢查停止信號
            if check_stop_signal(args.results_dir):
                print(f"\nStop signal detected! Stopping training.")
                break
            
            # 等待一段時間再進行下一次實驗
            if args.sleep_seconds > 0:
                print(f"Waiting {args.sleep_seconds} seconds before next experiment...")
                time.sleep(args.sleep_seconds)
                
    except KeyboardInterrupt:
        print(f"\n\nTraining stopped by user.")
        print(f"Total experiments completed: {experiment_count}")
    
    print(f"\nTraining session ended.")
    print(f"Total experiments: {experiment_count}")

if __name__ == "__main__":
    main()
