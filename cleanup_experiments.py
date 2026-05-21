#!/usr/bin/env python3
"""
實驗結果清理工具
用於清理舊的實驗結果，只保留必要的結果資料夾和文件

使用方法：
1. 清理單個實驗結果：
   python cleanup_experiments.py single /path/to/experiment/result

2. 清理多個實驗結果：
   python cleanup_experiments.py multiple /path/to/experiments/root

3. 清理特定模式的實驗結果：
   python cleanup_experiments.py pattern /path/to/experiments/root "Results_*_5shot_*"

4. 列出將被清理的內容（預覽模式）：
   python cleanup_experiments.py single /path/to/experiment/result --dry-run

5. 自定義要保留的資料夾：
   python cleanup_experiments.py single /path/to/experiment/result --keep "15_test_results_after_train_csv,metrics_output,logs,10_si_best_model"
"""

import argparse
import os
import shutil
import glob
import sys
from pathlib import Path

def cleanup_single_experiment(result_path, keep_folders=None, dry_run=False):
    """清理單個實驗結果
    
    Args:
        result_path: 實驗結果路徑
        keep_folders: 要保留的資料夾清單
        dry_run: 是否為預覽模式（不實際刪除）
    """
    if not os.path.exists(result_path):
        print(f"錯誤：路徑不存在 {result_path}")
        return False
    
    if not os.path.isdir(result_path):
        print(f"錯誤：不是資料夾 {result_path}")
        return False
    
    # 預設保留的資料夾
    default_keep = [
        '11_validation_results_in_train_csv',
        '15_test_results_after_train_csv',
        'logs',
        '10_si_best_model'  # 保留最佳模型
    ]
    
    keep_set = set(keep_folders) if keep_folders else set(default_keep)
    
    print(f"清理實驗結果：{result_path}")
    print(f"保留資料夾：{', '.join(sorted(keep_set))}")
    print(f"模式：{'預覽' if dry_run else '實際清理'}")
    print("-" * 60)
    
    removed_count = 0
    kept_count = 0
    
    try:
        for item in os.listdir(result_path):
            item_path = os.path.join(result_path, item)
            
            if os.path.isdir(item_path):
                if item in keep_set:
                    print(f"  ✓ 保留資料夾: {item}")
                    kept_count += 1
                else:
                    if dry_run:
                        print(f"  - 將刪除資料夾: {item}")
                    else:
                        try:
                            shutil.rmtree(item_path)
                            print(f"  ✓ 已刪除資料夾: {item}")
                        except Exception as e:
                            print(f"  ✗ 刪除失敗: {item} - {e}")
                    removed_count += 1
                    
            elif os.path.isfile(item_path):
                # 保留種子相關文件
                if item.lower() in {'seed.txt', 'seed_info.json', 'seeds.txt'}:
                    print(f"  ✓ 保留文件: {item}")
                    kept_count += 1
                else:
                    if dry_run:
                        print(f"  - 將刪除文件: {item}")
                    else:
                        try:
                            os.remove(item_path)
                            print(f"  ✓ 已刪除文件: {item}")
                        except Exception as e:
                            print(f"  ✗ 刪除失敗: {item} - {e}")
                    removed_count += 1
    
    except Exception as e:
        print(f"清理過程中發生錯誤: {e}")
        return False
    
    print("-" * 60)
    print(f"清理完成：保留 {kept_count} 項，{'將刪除' if dry_run else '已刪除'} {removed_count} 項")
    return True

def cleanup_multiple_experiments(root_path, keep_folders=None, dry_run=False):
    """清理多個實驗結果
    
    Args:
        root_path: 包含多個實驗結果的根目錄
        keep_folders: 要保留的資料夾清單
        dry_run: 是否為預覽模式
    """
    if not os.path.exists(root_path):
        print(f"錯誤：路徑不存在 {root_path}")
        return False
    
    # 尋找所有可能的實驗結果資料夾
    experiment_dirs = []
    
    # 尋找 Results_* 格式的資料夾
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path) and item.startswith('Results_'):
            experiment_dirs.append(item_path)
    
    if not experiment_dirs:
        print(f"在 {root_path} 中未找到實驗結果資料夾")
        return False
    
    print(f"找到 {len(experiment_dirs)} 個實驗結果資料夾")
    print("=" * 60)
    
    success_count = 0
    for exp_dir in sorted(experiment_dirs):
        print(f"\n處理: {os.path.basename(exp_dir)}")
        if cleanup_single_experiment(exp_dir, keep_folders, dry_run):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"批量清理完成：成功處理 {success_count}/{len(experiment_dirs)} 個實驗")
    return success_count == len(experiment_dirs)

def cleanup_by_pattern(root_path, pattern, keep_folders=None, dry_run=False):
    """根據模式清理實驗結果
    
    Args:
        root_path: 根目錄
        pattern: 匹配模式，如 "Results_*_5shot_*"
        keep_folders: 要保留的資料夾清單
        dry_run: 是否為預覽模式
    """
    if not os.path.exists(root_path):
        print(f"錯誤：路徑不存在 {root_path}")
        return False
    
    # 使用 glob 尋找匹配的資料夾
    search_pattern = os.path.join(root_path, pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        print(f"在 {root_path} 中未找到匹配模式 '{pattern}' 的資料夾")
        return False
    
    print(f"找到 {len(matching_dirs)} 個匹配的實驗結果資料夾")
    print("=" * 60)
    
    success_count = 0
    for exp_dir in sorted(matching_dirs):
        if os.path.isdir(exp_dir):
            print(f"\n處理: {os.path.basename(exp_dir)}")
            if cleanup_single_experiment(exp_dir, keep_folders, dry_run):
                success_count += 1
    
    print("\n" + "=" * 60)
    print(f"模式清理完成：成功處理 {success_count}/{len(matching_dirs)} 個實驗")
    return success_count == len(matching_dirs)

def main():
    parser = argparse.ArgumentParser(description='實驗結果清理工具')
    subparsers = parser.add_subparsers(dest='mode', help='清理模式')
    
    # 單個實驗清理
    single_parser = subparsers.add_parser('single', help='清理單個實驗結果')
    single_parser.add_argument('path', help='實驗結果路徑')
    single_parser.add_argument('--keep', type=str, help='要保留的資料夾，用逗號分隔')
    single_parser.add_argument('--dry-run', action='store_true', help='預覽模式，不實際刪除')
    
    # 多個實驗清理
    multiple_parser = subparsers.add_parser('multiple', help='清理多個實驗結果')
    multiple_parser.add_argument('root_path', help='包含多個實驗結果的根目錄')
    multiple_parser.add_argument('--keep', type=str, help='要保留的資料夾，用逗號分隔')
    multiple_parser.add_argument('--dry-run', action='store_true', help='預覽模式，不實際刪除')
    
    # 模式匹配清理
    pattern_parser = subparsers.add_parser('pattern', help='根據模式清理實驗結果')
    pattern_parser.add_argument('root_path', help='根目錄')
    pattern_parser.add_argument('pattern', help='匹配模式，如 "Results_*_5shot_*"')
    pattern_parser.add_argument('--keep', type=str, help='要保留的資料夾，用逗號分隔')
    pattern_parser.add_argument('--dry-run', action='store_true', help='預覽模式，不實際刪除')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # 處理 keep 參數
    keep_folders = None
    if args.keep:
        keep_folders = [folder.strip() for folder in args.keep.split(',') if folder.strip()]
    
    # 執行清理
    success = False
    if args.mode == 'single':
        success = cleanup_single_experiment(args.path, keep_folders, args.dry_run)
    elif args.mode == 'multiple':
        success = cleanup_multiple_experiments(args.root_path, keep_folders, args.dry_run)
    elif args.mode == 'pattern':
        success = cleanup_by_pattern(args.root_path, args.pattern, keep_folders, args.dry_run)
    
    if success:
        print("\n清理操作完成！")
    else:
        print("\n清理操作失敗！")
        sys.exit(1)

if __name__ == "__main__":
    main()
