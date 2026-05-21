#!/usr/bin/env python3
"""
準確率監控腳本
持續監控實驗結果，當達到目標準確率時停止訓練

使用方法:
python monitor_accuracy.py "results" --target 0.9 --dataset split4 --shots 5
"""

import argparse
import os
import time
import pandas as pd
import glob
from pathlib import Path
import json

def find_latest_experiment(results_dir, dataset, shots):
    """找到最新的實驗結果"""
    pattern = f"Results_{dataset}_{shots}shot_*"
    search_path = os.path.join(results_dir, pattern)
    experiment_dirs = glob.glob(search_path)
    
    if not experiment_dirs:
        return None
    
    # 按修改時間排序，取最新的
    latest_dir = max(experiment_dirs, key=os.path.getmtime)
    return latest_dir

def read_accuracy_from_experiment(experiment_dir, target_ways=5):
    """從實驗結果中讀取準確率"""
    try:
        # 優先讀取 5ways 或 10ways 的詳細結果
        ways_csv = os.path.join(experiment_dir, "15_test_results_after_train_csv", "pr_data", f"{target_ways}ways", f"{target_ways}ways_per_case_metrics.csv")
        if os.path.exists(ways_csv):
            df = pd.read_csv(ways_csv)
            if 'accuracy' in df.columns:
                accuracy = df['accuracy'].mean()
                return accuracy
        
        # 備用方案：讀取 avg.csv
        avg_csv = os.path.join(experiment_dir, "15_test_results_after_train_csv", "avg.csv")
        if os.path.exists(avg_csv):
            df = pd.read_csv(avg_csv)
            # 計算所有數值列的平均值
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                accuracy = df[numeric_cols].mean().mean()
                return accuracy
        
        # 最後備用：讀取 results.csv
        results_csv = os.path.join(experiment_dir, "15_test_results_after_train_csv", "results.csv")
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                accuracy = df[numeric_cols].mean().mean()
                return accuracy
                
    except Exception as e:
        print(f"Error reading accuracy from {experiment_dir}: {e}")
    
    return None

def save_stop_signal(results_dir):
    """保存停止信號文件"""
    stop_file = os.path.join(results_dir, "STOP_TRAINING.txt")
    with open(stop_file, 'w') as f:
        f.write(f"Target accuracy reached at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Stop signal saved to: {stop_file}")

def check_stop_signal(results_dir):
    """檢查是否有停止信號"""
    stop_file = os.path.join(results_dir, "STOP_TRAINING.txt")
    return os.path.exists(stop_file)

def main():
    parser = argparse.ArgumentParser(description='Monitor experiment accuracy')
    parser.add_argument('results_dir', help='Results directory path')
    parser.add_argument('--target', type=float, required=True, help='Target accuracy (e.g., 0.9)')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., split4)')
    parser.add_argument('--shots', type=int, required=True, help='Number of shots (e.g., 5)')
    parser.add_argument('--ways', type=int, default=5, help='Number of ways to monitor (5 or 10)')
    parser.add_argument('--check_interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--max_wait', type=int, default=43200, help='Maximum wait time in seconds (default: 12 hours)')
    
    args = parser.parse_args()
    
    print(f"Monitoring accuracy for {args.dataset} {args.shots}shot {args.ways}ways")
    print(f"Target accuracy: {args.target}")
    print(f"Results directory: {args.results_dir}")
    print(f"Check interval: {args.check_interval} seconds")
    print("=" * 60)
    
    start_time = time.time()
    best_accuracy = 0.0
    best_experiment = None
    
    while True:
        # 檢查是否已經有停止信號
        if check_stop_signal(args.results_dir):
            print("Stop signal detected. Exiting monitor.")
            break
        
        # 檢查是否超時
        elapsed = time.time() - start_time
        if elapsed > args.max_wait:
            print(f"Timeout reached ({args.max_wait}s). Exiting monitor.")
            break
        
        # 尋找最新實驗
        latest_experiment = find_latest_experiment(args.results_dir, args.dataset, args.shots)
        
        if latest_experiment:
            accuracy = read_accuracy_from_experiment(latest_experiment, args.ways)
            
            if accuracy is not None:
                print(f"[{time.strftime('%H:%M:%S')}] Latest experiment: {os.path.basename(latest_experiment)}")
                print(f"  Accuracy: {accuracy:.6f}")
                
                # 更新最佳結果
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_experiment = latest_experiment
                    print(f"  New best accuracy: {best_accuracy:.6f}")
                
                # 檢查是否達到目標
                if accuracy >= args.target:
                    print(f"\n🎉 TARGET REACHED! 🎉")
                    print(f"Accuracy: {accuracy:.6f} >= {args.target}")
                    print(f"Experiment: {latest_experiment}")
                    save_stop_signal(args.results_dir)
                    break
                else:
                    print(f"  Target: {args.target:.6f} (need {args.target - accuracy:.6f} more)")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No accuracy data found in latest experiment")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No experiments found yet...")
        
        print(f"  Best so far: {best_accuracy:.6f}")
        print(f"  Elapsed: {elapsed/60:.1f} minutes")
        print("-" * 40)
        
        time.sleep(args.check_interval)
    
    print(f"\nFinal best accuracy: {best_accuracy:.6f}")
    if best_experiment:
        print(f"Best experiment: {best_experiment}")

if __name__ == "__main__":
    main()
