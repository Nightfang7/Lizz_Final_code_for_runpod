#!/usr/bin/env python3
"""
快速統整腳本 - 簡化版本
quick_aggregate.py

快速統整實驗結果，生成簡潔的CSV表格

使用方法:
python quick_aggregate.py "workspace/results_splt3"
"""

import os
import pandas as pd
import re
import sys
import json
from pathlib import Path

def extract_info_from_folder(folder_name):
    """從資料夾名稱提取信息"""
    # 更靈活的正則表達式，支持包含下劃線的資料集名稱
    patterns = [
        r'Results_([^_]+)_(\d+)shot_(?:seed(\d+)|run(\d+))',  # 原始模式
        r'Results_([^_]+(?:_[^_]+)*)_(\d+)shot_(?:seed(\d+)|run(\d+))',  # 支持多個下劃線
        r'Results_(.+?)_(\d+)shot_(?:seed(\d+)|run(\d+))'  # 最寬鬆的模式
    ]
    
    for pattern in patterns:
        match = re.match(pattern, folder_name)
        if match:
            return {
                'dataset': match.group(1),
                'shots': int(match.group(2)),
                'seed_from_name': int(match.group(3)) if match.group(3) else None,
                'run_id': int(match.group(4)) if match.group(4) else None
            }
    
    # 如果都不匹配，嘗試手動解析
    if folder_name.startswith('Results_') and '_shot_' in folder_name:
        parts = folder_name.split('_')
        if len(parts) >= 4:
            # 找到 shot 的位置
            shot_idx = None
            for i, part in enumerate(parts):
                if part == 'shot':
                    shot_idx = i
                    break
            
            if shot_idx and shot_idx > 1:
                dataset = '_'.join(parts[1:shot_idx])
                shots = int(parts[shot_idx-1])
                
                # 檢查是否有 seed 或 run
                if shot_idx + 2 < len(parts) and parts[shot_idx + 1] in ['seed', 'run']:
                    seed_or_run = int(parts[shot_idx + 2])
                    return {
                        'dataset': dataset,
                        'shots': shots,
                        'seed_from_name': seed_or_run if parts[shot_idx + 1] == 'seed' else None,
                        'run_id': seed_or_run if parts[shot_idx + 1] == 'run' else None
                    }
    
    return None

def get_actual_seed(experiment_folder):
    """獲取實際使用的seed"""
    # 嘗試從 seed_info.json 讀取
    seed_info_path = experiment_folder / 'seed_info.json'
    if seed_info_path.exists():
        try:
            with open(seed_info_path, 'r') as f:
                return json.load(f)['random_seed']
        except:
            pass
    
    # 嘗試從 SEED.txt 讀取
    seed_txt_path = experiment_folder / 'SEED.txt'
    if seed_txt_path.exists():
        try:
            with open(seed_txt_path, 'r') as f:
                for line in f:
                    if 'Random Seed:' in line:
                        return int(line.split(':')[1].strip())
        except:
            pass
    
    return None

def process_metrics_csv(csv_path):
    """處理單個metrics CSV文件"""
    try:
        df = pd.read_csv(csv_path)
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'avg_pr_auc']
        return df[metrics].mean().to_dict()
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_aggregate.py <results_directory>")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist!")
        sys.exit(1)
    
    results = []
    
    # 掃描所有實驗資料夾
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
        
        print(f"Processing folder: {folder.name}")
        info = extract_info_from_folder(folder.name)
        if not info:
            print(f"  -> Could not parse folder name: {folder.name}")
            continue
        
        print(f"  -> Parsed: dataset={info['dataset']}, shots={info['shots']}, seed={info['seed_from_name']}, run_id={info['run_id']}")
        
        # 獲取實際seed
        actual_seed = get_actual_seed(folder)
        if not actual_seed:
            actual_seed = info['seed_from_name']  # 備用方案
        
        # 處理5ways和10ways的結果
        for ways in [5, 10]:
            csv_path = folder / "15_test_results_after_train_csv" / "pr_data" / f"{ways}ways" / f"{ways}ways_per_case_metrics.csv"
            
            if csv_path:
                print(f"Found CSV: {csv_path}")
                metrics = process_metrics_csv(csv_path)
                if metrics:
                    result = {
                        'dataset': info['dataset'],
                        'shots': info['shots'],
                        'ways': ways,
                        'seed': actual_seed,
                        'run_id': info['run_id'],
                        **metrics
                    }
                    results.append(result)
                    print(f"Processed: {info['dataset']} {info['shots']}shot {ways}ways - {metrics}")
                else:
                    print(f"Failed to process metrics from: {csv_path}")
            else:
                print(f"No CSV found for {info['dataset']} {info['shots']}shot {ways}ways")
    
    if not results:
        print("No results found!")
        return
    
    # 轉換為DataFrame
    df = pd.DataFrame(results)
    
    # 按dataset, shots, ways分組並生成結果
    for (dataset, shots, ways), group in df.groupby(['dataset', 'shots', 'ways']):
        print(f"\n=== {dataset.upper()} {shots}shot {ways}ways ===")
        
        # 排序並顯示詳細結果
        group_sorted = group.sort_values('seed')
        detail_cols = ['seed', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'avg_pr_auc']
        
        print("詳細結果:")
        print(group_sorted[detail_cols].round(4).to_string(index=False))
        
        # 計算統計摘要
        metrics_cols = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'avg_pr_auc']
        stats = group[metrics_cols].agg(['count', 'mean', 'std', 'min', 'max'])
        
        print(f"\n統計摘要 (共{len(group)}個實驗):")
        print(stats.round(4).to_string())
        
        # 保存CSV
        filename = f"{dataset}_{shots}shot_{ways}ways_aggregated.csv"
        group_sorted[detail_cols].to_csv(filename, index=False, float_format='%.4f')
        print(f"已保存: {filename}")
        
        # 保存統計摘要
        summary_filename = f"{dataset}_{shots}shot_{ways}ways_summary.csv"
        stats.to_csv(summary_filename, float_format='%.4f')
        print(f"已保存: {summary_filename}")

if __name__ == "__main__":
    main()