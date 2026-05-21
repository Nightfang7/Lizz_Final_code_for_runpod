#!/usr/bin/env python3
"""
統一實驗執行腳本 - 支持多種啟動模式
使用方法：
1. 單個實驗：python run_experiments.py single split3 5 456
2. 批量實驗：python run_experiments.py batch quick_test
3. 自定義批量：python run_experiments.py custom "split3 5 456,split3 10 123"
4. Unix 時間種子實驗：python run_experiments.py unix_seed split3 5 3  # 跑3次
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import os
import sys
import time
from datetime import datetime
import subprocess
import shutil
import logging
import json
import signal
import psutil
import gc

from experiment_config import ExperimentConfig, EXPERIMENT_BATCHES

class ExperimentRunner:
    def __init__(self, environment='local', hparam_overrides=None):
        self.config = ExperimentConfig(environment)
        # Apply hyperparameter overrides if provided
        if isinstance(hparam_overrides, dict) and hparam_overrides:
            try:
                self.config.experiment_params.update(hparam_overrides)
            except Exception:
                pass

    def cleanup_intermediate_artifacts(self, result_path, keep=['15_test_results_after_train_csv', 'metrics_output', 'logs']):
        """刪除單次實驗中的中間產物，只保留結果與日志。

        keep: 要保留的資料夾名稱清單（直接位於 result_path 下）
        """
        try:
            keep_set = set(keep)
            for item in os.listdir(result_path):
                p = os.path.join(result_path, item)
                if os.path.isdir(p) and item not in keep_set:
                    try:
                        shutil.rmtree(p)
                        print(f"  - Removed folder: {p}")
                    except Exception as e:
                        print(f"  - Skip remove {p}: {e}")
                elif os.path.isfile(p):
                    # 一般保留 seed 記錄與配置
                    if item.lower() not in {'seed.txt', 'seed_info.json'}:
                        try:
                            os.remove(p)
                            print(f"  - Removed file: {p}")
                        except Exception as e:
                            print(f"  - Skip remove file {p}: {e}")
        except Exception as e:
            print(f"Cleanup warning: {e}")
        
    def setup_logging(self, result_path, dataset, shots, random_seed, run_id=None):
        """設置logging系統"""
        log_dir = os.path.join(result_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 設置log文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_id is not None:
            log_filename = f"experiment_{dataset}_{shots}shot_run{run_id:03d}_seed{random_seed}_{timestamp}.log"
        else:
            log_filename = f"experiment_{dataset}_{shots}shot_seed{random_seed}_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        
        # 配置logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger('experiment')
        
        # 記錄實驗開始信息
        logger.info("="*60)
        logger.info(f"EXPERIMENT STARTED")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Shots: {shots}")
        logger.info(f"Random Seed: {random_seed}")
        if run_id is not None:
            logger.info(f"Run ID: {run_id:03d}")
        logger.info(f"Result Path: {result_path}")
        logger.info(f"Log File: {log_filepath}")
        logger.info("="*60)
        
        return logger, log_filepath
        
    def setup_experiment_folder(self, dataset_name, shots, random_seed, run_id=None):
        """設置實驗資料夾"""
        result_folder_name = self.config.get_result_folder_name(dataset_name, shots, random_seed, run_id)
        result_path = os.path.join(self.config.base_paths['result_base'], result_folder_name)
        
        # 創建結果資料夾
        os.makedirs(result_path, exist_ok=True)
        
        # 複製資料夾結構模板
        template_path = self.config.base_paths['template_dir']
        if os.path.exists(template_path):
            self._copy_folder_structure(template_path, result_path)
        else:
            self._create_basic_structure(result_path)
        
        # 創建 seed 記錄文件
        self._save_seed_info(result_path, random_seed, run_id)
        
        return result_path
    
    def _save_seed_info(self, result_path, random_seed, run_id=None):
        """保存隨機種子信息到文件"""
        seed_info = {
            'random_seed': random_seed,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'run_id': run_id
        }
        
        seed_file_path = os.path.join(result_path, 'seed_info.json')
        with open(seed_file_path, 'w', encoding='utf-8') as f:
            json.dump(seed_info, f, indent=2, ensure_ascii=False)
        
        # 也創建一個簡單的文本文件方便快速查看
        seed_txt_path = os.path.join(result_path, 'SEED.txt')
        with open(seed_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Random Seed: {random_seed}\n")
            f.write(f"Timestamp: {seed_info['timestamp']}\n")
            f.write(f"DateTime: {seed_info['datetime']}\n")
            if run_id is not None:
                f.write(f"Run ID: {run_id:03d}\n")
    
    def _copy_folder_structure(self, template_dir, target_dir):
        """複製資料夾結構"""
        for item in os.listdir(template_dir):
            src = os.path.join(template_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    
    def _create_basic_structure(self, base_dir):
        """創建基本資料夾結構"""
        folders = [
            '1_autoencoder_model', '2_autoencoder_img', '3_train_clustered_inter_dest',
            '4_train_clustered_best_dest', '5_train_prototype', '6_regrouped_train',
            '7_test_clustered_inter_dest', '8_test_clustered_best_dest', '9_si_inter_model',
            '10_si_best_model', '11_validation_results_in_train_csv', '12_test_prototype_dest',
            '13_train_proto_clustered_inter_dest', '14_train_proto_clustered_best_dest',
            '15_test_results_after_train_csv', 'metrics_output', 'logs'
        ]
        for folder in folders:
            os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
    
    def run_single_experiment(self, dataset, shots, random_seed, run_id=None):
        """執行單個實驗"""
        
        # 如果 random_seed 是 'unix' 字符串，生成當前時間戳
        if random_seed == 'unix':
            actual_seed = self.config.generate_unix_timestamp_seed()
            print(f"Generated Unix timestamp seed: {actual_seed}")
            time.sleep(0.1)  # 確保不同實驗間的種子不同
        else:
            actual_seed = random_seed
        
        # 設置實驗資料夾
        result_path = self.setup_experiment_folder(dataset, shots, actual_seed, run_id)
        
        # 設置logging
        logger, log_filepath = self.setup_logging(result_path, dataset, shots, actual_seed, run_id)
        
        # 設置信號處理器來捕獲 segmentation fault
        def signal_handler(signum, frame):
            logger.error(f"收到信號 {signum}，實驗被中斷")
            raise KeyboardInterrupt("實驗被信號中斷")
        
        # 註冊信號處理器
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            run_info = f"run{run_id:03d}_" if run_id is not None else ""
            logger.info(f"Starting experiment: {dataset}_{shots}shot_{run_info}seed{actual_seed}")
            
            # 獲取資料集路徑並驗證
            dataset_paths = self.config.get_dataset_paths(dataset, shots)
            logger.info("Validating dataset paths...")
            self.config.validate_paths(dataset_paths)
            
            # 導入並執行訓練函數
            logger.info("Importing training modules...")
            from main_loop import complete_training_loop_1
            import torchvision.transforms as transforms
            import torch
            import numpy as np
            import random
            
            # 釋放前一次可能殘留的 GPU 資源，並安全設定 random seed
            logger.info(f"Setting random seed to: {actual_seed}")
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                random.seed(actual_seed)
                np.random.seed(actual_seed)
                # torch.manual_seed 在某些版本可能觸發 CUDA 相關初始化，這裡加上保護
                try:
                    torch.manual_seed(actual_seed)
                except Exception as e_seed:
                    logger.error(f"torch.manual_seed failed: {e_seed}")
                    raise
                if torch.cuda.is_available():
                    try:
                        torch.cuda.manual_seed_all(actual_seed)
                        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
                    except Exception as e_cuda_seed:
                        logger.error(f"torch.cuda.manual_seed_all failed: {e_cuda_seed}")
                        raise
                else:
                    logger.info("CUDA not available, using CPU")
            except Exception as seed_err:
                # 讓上層捕獲並將此 run 標記為失敗，避免卡住整體流程
                logger.error(f"Random seeding failed due to CUDA/driver state: {seed_err}")
                raise
            
            # 準備變換
            transformation = transforms.Compose([
                transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
            
            # 構建完整配置
            logger.info("Building experiment configuration...")
            full_config = {
                'autoencoder_training_data_path': dataset_paths['autoencoder_training_data_path'],
                'autoencoder_save_model_path': os.path.join(result_path, '1_autoencoder_model', 'autoencoder_model.pth'),
                'autoencoder_save_img_base_path': os.path.join(result_path, '2_autoencoder_img'),
                'training_data_cluster_target': dataset_paths['training_data_cluster_target'],
                'training_clustered_inter_dir': os.path.join(result_path, '3_train_clustered_inter_dest'),
                'training_clustered_best_dir': os.path.join(result_path, '4_train_clustered_best_dest'),
                'train_prototype_dir': os.path.join(result_path, '5_train_prototype'),
                'proto_clustered_best_dest_dir': os.path.join(result_path, '14_train_proto_clustered_best_dest'),
                'proto_clustered_inter_dest_dir': os.path.join(result_path, '13_train_proto_clustered_inter_dest'),
                'regrouped_target_dir': os.path.join(result_path, '6_regrouped_train'),
                'test_cluster_target_dir': dataset_paths['proto_data_dir'],
                'testing_clustered_inter_dir': os.path.join(result_path, '7_test_clustered_inter_dest'),
                'testing_clustered_best_dir': os.path.join(result_path, '8_test_clustered_best_dest'),
                'save_inter_si_model_dir': os.path.join(result_path, '9_si_inter_model'),
                'save_best_si_model_dir': os.path.join(result_path, '10_si_best_model'),
                'vali_data_dir': dataset_paths['testing_data_dir'],
                'csv_save_dir': os.path.join(result_path, '11_validation_results_in_train_csv'),
                'test_prototype_dest_dir': os.path.join(result_path, '12_test_prototype_dest'),
                'result_csv_dir': os.path.join(result_path, '15_test_results_after_train_csv'),
                'transformation': transformation,
                'random_seed': actual_seed,
                'encoder_path': os.path.join(result_path, '1_autoencoder_model', 'autoencoder_model.pth'),
                'shots': shots,
                **self.config.experiment_params
            }
            
            logger.info(f"Configuration complete. Starting training...")
            logger.info(f"Key parameters:")
            logger.info(f"  - Network: {full_config['net_name']}")
            logger.info(f"  - Epochs: {full_config['si_epoch_num']}")
            logger.info(f"  - Ways: {full_config['way_list']}")
            logger.info(f"  - Cases: {full_config['case_len']}")
            
            # 監控記憶體使用
            def log_memory_usage():
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    logger.info(f"記憶體使用: {memory_mb:.1f} MB")
                    if memory_mb > 8000:  # 超過 8GB 警告
                        logger.warning(f"記憶體使用過高: {memory_mb:.1f} MB")
                except Exception:
                    pass
            
            log_memory_usage()
            
            # 重定向stdout到log文件
            original_stdout = sys.stdout
            
            class TeeOutput:
                def __init__(self, file, terminal):
                    self.file = file
                    self.terminal = terminal
                
                def write(self, message):
                    self.file.write(message)
                    self.terminal.write(message)
                    self.file.flush()
                
                def flush(self):
                    self.file.flush()
                    self.terminal.flush()
            
            # 創建log文件的寫入器
            with open(log_filepath, 'a', encoding='utf-8') as log_file:
                sys.stdout = TeeOutput(log_file, original_stdout)
                
                try:
                    # 執行訓練
                    logger.info("Starting complete_training_loop_1...")
                    log_memory_usage()
                    complete_training_loop_1(**full_config)
                    log_memory_usage()
                    
                finally:
                    # 恢復原始輸出
                    sys.stdout = original_stdout
                    # 強制垃圾回收
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            logger.info("="*60)
            logger.info(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
            logger.info(f"Results saved to: {result_path}")
            logger.info(f"Log saved to: {log_filepath}")
            logger.info(f"Actual seed used: {actual_seed}")
            logger.info("="*60)
            
            return True, result_path, actual_seed
            
        except KeyboardInterrupt as e:
            logger.error("="*60)
            logger.error(f"EXPERIMENT INTERRUPTED!")
            logger.error(f"Dataset: {dataset}, Shots: {shots}, Seed: {actual_seed}")
            if run_id is not None:
                logger.error(f"Run ID: {run_id:03d}")
            logger.error(f"Interrupt reason: {str(e)}")
            logger.error("="*60)
            return False, None, actual_seed
        except Exception as e:
            logger.error("="*60)
            logger.error(f"EXPERIMENT FAILED!")
            logger.error(f"Dataset: {dataset}, Shots: {shots}, Seed: {actual_seed}")
            if run_id is not None:
                logger.error(f"Run ID: {run_id:03d}")
            logger.error(f"Error: {str(e)}")
            logger.error("="*60)

            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())

            # 清除失敗的殘留資料夾，避免下次重跑時混入舊資料
            try:
                if result_path and os.path.exists(result_path):
                    shutil.rmtree(result_path)
                    print(f"Cleaned up failed experiment folder: {result_path}")
            except Exception:
                pass

            return False, None, actual_seed
        finally:
            # 恢復原始信號處理器
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            # 強制清理資源
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
    
    def run_unix_seed_experiments(self, dataset, shots, repeat_count, cleanup=False, keep_folders=None):
        """執行多次使用Unix時間戳作為種子的實驗

        cleanup: 是否在每次實驗後清理中間產物
        keep_folders: 要保留的資料夾清單（位於單次結果根目錄下）
        """
        print(f"Running {repeat_count} experiments with Unix timestamp seeds")
        print(f"Dataset: {dataset}, Shots: {shots}")
        
        successful = []
        failed = []
        seeds_used = []
        start_time = time.time()
        per_run_summaries = []
        
        for run_id in range(1, repeat_count + 1):
            print(f"\n[{run_id}/{repeat_count}] Running Unix seed experiment...")
            # 在每次 run 之前嘗試釋放 GPU 快取，避免前一次失敗影響下一次
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            success, result_path, actual_seed = self.run_single_experiment(dataset, shots, 'unix', run_id)
            exp_name = f"{dataset}_{shots}shot_run{run_id:03d}"
            seeds_used.append(actual_seed)
            
            if success:
                successful.append((exp_name, result_path, actual_seed))
                # Collect metrics for this run
                avg_csv = os.path.join(result_path, '15_test_results_after_train_csv', 'avg.csv')
                precision_avg_csv = os.path.join(result_path, '15_test_results_after_train_csv', 'precision_avg.csv')
                recall_avg_csv = os.path.join(result_path, '15_test_results_after_train_csv', 'recall_avg.csv')
                f1_avg_csv = os.path.join(result_path, '15_test_results_after_train_csv', 'f1_avg.csv')
                summary = {
                    'exp_name': exp_name,
                    'dataset': dataset,
                    'shots': shots,
                    'seed': actual_seed,
                    'result_path': result_path,
                    'timestamp': datetime.now().isoformat(),
                    'hparams': self.config.experiment_params.copy()
                }
                # Load accuracy averages
                try:
                    import pandas as pd
                    if os.path.exists(avg_csv):
                        df = pd.read_csv(avg_csv)
                        # Overall accuracy: mean of all numeric cells
                        acc = float(pd.to_numeric(df.select_dtypes(include=['number']).values.flatten(), errors='coerce').mean())
                        summary['accuracy_mean'] = acc
                        # Per-column (way) values
                        for col in df.columns:
                            try:
                                summary[f'accuracy_{col}'] = float(pd.to_numeric(df[col], errors='coerce').mean())
                            except Exception:
                                pass
                except Exception:
                    pass
                # Load PRF macro averages if available
                try:
                    import pandas as pd
                    if os.path.exists(precision_avg_csv):
                        p_df = pd.read_csv(precision_avg_csv)
                        summary['precision_macro_mean'] = float(pd.to_numeric(p_df.select_dtypes(include=['number']).values.flatten(), errors='coerce').mean())
                    if os.path.exists(recall_avg_csv):
                        r_df = pd.read_csv(recall_avg_csv)
                        summary['recall_macro_mean'] = float(pd.to_numeric(r_df.select_dtypes(include=['number']).values.flatten(), errors='coerce').mean())
                    if os.path.exists(f1_avg_csv):
                        f_df = pd.read_csv(f1_avg_csv)
                        summary['f1_macro_mean'] = float(pd.to_numeric(f_df.select_dtypes(include=['number']).values.flatten(), errors='coerce').mean())
                except Exception:
                    pass
                # Persist per-run summary JSON
                try:
                    run_summary_path = os.path.join(result_path, 'metrics_output', 'run_summary.json')
                    os.makedirs(os.path.dirname(run_summary_path), exist_ok=True)
                    with open(run_summary_path, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                per_run_summaries.append(summary)
                # Cleanup per run if requested
                if cleanup and result_path:
                    try:
                        self.cleanup_intermediate_artifacts(
                            result_path,
                            keep=keep_folders or ['11_validation_results_in_train_csv','15_test_results_after_train_csv','logs']
                        )
                    except Exception as _e:
                        print(f"Cleanup error (ignored): {_e}")
            else:
                failed.append((exp_name, actual_seed))
            
            # 在實驗間稍作停頓，確保時間戳不同
            if run_id < repeat_count:
                time.sleep(1)
                # 強制清理記憶體
                try:
                    import torch
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
        
        # 總結報告
        total_duration = time.time() - start_time
        print(f"\n{'='*60}")
        print("UNIX SEED EXPERIMENTS SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {dataset}, Shots: {shots}")
        print(f"Total experiments: {repeat_count}")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Successful: {len(successful)}/{repeat_count}")
        print(f"Failed: {len(failed)}/{repeat_count}")
        print(f"Seeds used: {seeds_used}")
        
        if successful:
            print(f"\nSuccessful experiments:")
            for exp_name, result_path, seed in successful:
                print(f"  ✓ {exp_name} (seed: {seed}) -> {result_path}")
        
        if failed:
            print(f"\nFailed experiments:")
            for exp_name, seed in failed:
                print(f"  ✗ {exp_name} (seed: {seed})")

        # Aggregate and save all run summaries
        if per_run_summaries:
            try:
                import pandas as pd
                summary_df = pd.DataFrame(per_run_summaries)
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                aggregate_dir = os.path.join(self.config.base_paths['result_base'], 'unix_seed_summaries')
                os.makedirs(aggregate_dir, exist_ok=True)
                base_name = f"summary_{dataset}_{shots}shot_{stamp}"
                csv_path = os.path.join(aggregate_dir, base_name + '.csv')
                json_path = os.path.join(aggregate_dir, base_name + '.json')
                summary_df.to_csv(csv_path, index=False)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(per_run_summaries, f, indent=2, ensure_ascii=False)
                print(f"\nSaved aggregated unix-seed summary:")
                print(f"  CSV:  {csv_path}")
                print(f"  JSON: {json_path}")
            except Exception as e:
                print(f"Warning: failed to save aggregated summary: {e}")
    
    def run_batch_experiments(self, batch_name, cleanup=False, keep_folders=None):
        """執行批量實驗

        cleanup: 是否在每次實驗後清理中間產物
        keep_folders: 要保留的資料夾清單
        """
        if batch_name not in EXPERIMENT_BATCHES:
            print(f"Unknown batch: {batch_name}")
            print(f"Available batches: {list(EXPERIMENT_BATCHES.keys())}")
            return
        
        experiments = EXPERIMENT_BATCHES[batch_name]
        print(f"Running batch: {batch_name}")
        
        # 檢查是否為 Unix seed 批量實驗
        if experiments and len(experiments[0]) == 4 and experiments[0][2] == 'unix':
            # Unix seed 批量實驗格式: (dataset, shots, 'unix', repeat_count)
            for dataset, shots, _, repeat_count in experiments:
                self.run_unix_seed_experiments(dataset, shots, repeat_count)
            return
        
        # 常規批量實驗
        print(f"Total experiments: {len(experiments)}")
        
        successful = []
        failed = []
        start_time = time.time()
        
        for i, (dataset, shots, random_seed) in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Running experiment...")
            
            success, result_path, actual_seed = self.run_single_experiment(dataset, shots, random_seed)
            exp_name = f"{dataset}_{shots}shot_seed{actual_seed}"
            
            if success:
                successful.append((exp_name, result_path))
                if cleanup and result_path:
                    try:
                        self.cleanup_intermediate_artifacts(
                            result_path,
                            keep=keep_folders or ['11_validation_results_in_train_csv','15_test_results_after_train_csv','logs']
                        )
                    except Exception as _e:
                        print(f"Cleanup error (ignored): {_e}")
            else:
                failed.append(exp_name)
        
        # 總結報告
        total_duration = time.time() - start_time
        print(f"\n{'='*60}")
        print("BATCH EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {total_duration/3600:.2f} hours")
        print(f"Successful: {len(successful)}/{len(experiments)}")
        print(f"Failed: {len(failed)}/{len(experiments)}")
        
        if successful:
            print(f"\nSuccessful experiments:")
            for exp_name, result_path in successful:
                print(f"  ✓ {exp_name} -> {result_path}")
        
        if failed:
            print(f"\nFailed experiments:")
            for exp_name in failed:
                print(f"  ✗ {exp_name}")

    def _read_metric_from_result(self, result_path, metric_key):
        """從單次實驗結果中讀取指定指標的平均值。

        metric_key 可選: 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'
        """
        try:
            import pandas as pd
            metrics_dir = os.path.join(result_path, '15_test_results_after_train_csv')
            file_map = {
                'accuracy': 'avg.csv',
                'precision_macro': 'precision_avg.csv',
                'recall_macro': 'recall_avg.csv',
                'f1_macro': 'f1_avg.csv',
            }
            fname = file_map.get(metric_key)
            if not fname:
                return None
            csv_path = os.path.join(metrics_dir, fname)
            if not os.path.exists(csv_path):
                return None
            df = pd.read_csv(csv_path)
            vals = pd.to_numeric(df.select_dtypes(include=['number']).values.flatten(), errors='coerce')
            if vals.size == 0:
                return None
            return float(pd.Series(vals).mean())
        except Exception:
            return None

    def run_until_threshold(self, dataset, shots, threshold_metric, threshold_value, max_runs=None, sleep_seconds=1, cleanup=False, keep_folders=None):
        """反覆以 Unix 時間戳為種子執行實驗，直到達到門檻或到達最大次數。

        cleanup: 是否每次成功完成後清理中間產物
        keep_folders: 要保留的資料夾清單
        """
        print(f"Run-until-threshold started: dataset={dataset}, shots={shots}, metric={threshold_metric}, threshold={threshold_value}")
        run_id = 1
        best_metric = None
        best_run = None
        best_result_path = None
        start_time = time.time()

        while True:
            print(f"\n[run {run_id}] Starting...")
            # 在每次 run 之前嘗試釋放 GPU 快取
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            success, result_path, actual_seed = self.run_single_experiment(dataset, shots, 'unix', run_id)
            if not success:
                print(f"[run {run_id}] Failed. Continue to next.")
            else:
                metric_value = self._read_metric_from_result(result_path, threshold_metric)
                if metric_value is None:
                    print(f"[run {run_id}] Warning: could not read metric '{threshold_metric}'. Continue.")
                else:
                    print(f"[run {run_id}] {threshold_metric} = {metric_value:.6f}")
                    if best_metric is None or metric_value > best_metric:
                        best_metric = metric_value
                        best_run = run_id
                        best_result_path = result_path
                    # Cleanup current run if requested (and not best or regardless?)
                    if cleanup and result_path:
                        try:
                            self.cleanup_intermediate_artifacts(
                                result_path,
                                keep=keep_folders or ['11_validation_results_in_train_csv','15_test_results_after_train_csv','logs']
                            )
                        except Exception as _e:
                            print(f"Cleanup error (ignored): {_e}")
                    if metric_value >= threshold_value:
                        dur = time.time() - start_time
                        print(f"\nReach threshold! run {run_id} achieved {threshold_metric}={metric_value:.6f} >= {threshold_value}")
                        print(f"Result path: {result_path}")
                        print(f"Total time: {dur/3600:.2f} hours")
                        return True, result_path, actual_seed, metric_value
            if max_runs is not None and run_id >= max_runs:
                dur = time.time() - start_time
                print(f"\nStopped after max_runs={max_runs}. Best {threshold_metric}={best_metric} at run {best_run} -> {best_result_path}")
                print(f"Total time: {dur/3600:.2f} hours")
                return False, best_result_path, None, best_metric
            run_id += 1
            if sleep_seconds and run_id > 1:
                time.sleep(sleep_seconds)
                # 強制清理記憶體
                try:
                    import torch
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

def main():
    parser = argparse.ArgumentParser(description='Unified Experiment Runner')
    subparsers = parser.add_subparsers(dest='mode', help='Running mode')
    
    # 單個實驗模式
    single_parser = subparsers.add_parser('single', help='Run single experiment')
    single_parser.add_argument('dataset', choices=['split3', 'split4', 'apidms', 'apidms_regroup','split4_10%noise', 'split4_20%noise', 'split4_50%noise', 'apidms_10%noise', 'apidms_20%noise', 'apidms_50%noise', 'win10_split4', 'plusRed_00177', 'plusRed_00177_fulltime', 'vs0177_full_split2-1', 'ka_malBazzar'])  # 新增 split4
    single_parser.add_argument('shots', type=int, choices=[5, 10])
    single_parser.add_argument('random_seed', help='Random seed (number or "unix" for timestamp)')
    single_parser.add_argument('--cleanup', action='store_true', help='Cleanup intermediate artifacts after the run')
    single_parser.add_argument('--keep_folders', type=str, help='Comma-separated folder names to keep (default: 15_test_results_after_train_csv,metrics_output,logs)')
    
    # Unix 時間種子實驗模式
    unix_parser = subparsers.add_parser('unix_seed', help='Run multiple experiments with Unix timestamp seeds')
    unix_parser.add_argument('dataset', choices=['split3', 'split4', 'apidms', 'apidms_regroup','split4_10%noise', 'split4_20%noise', 'split4_50%noise', 'apidms_10%noise', 'apidms_20%noise', 'apidms_50%noise', 'win10_split4', 'plusRed_00177', 'plusRed_00177_fulltime', 'vs0177_full_split2-1', 'ka_malBazzar'])
    unix_parser.add_argument('shots', type=int, choices=[5, 10])
    unix_parser.add_argument('repeat_count', type=int, help='Number of experiments to run')
    unix_parser.add_argument('--threshold_metric', default='accuracy', choices=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], help='Metric to check threshold')
    unix_parser.add_argument('--threshold_value', type=float, help='Threshold value to check against')
    unix_parser.add_argument('--hparam_config', type=str, help='Path to JSON with hyperparameter overrides')
    unix_parser.add_argument('--cleanup', action='store_true', help='Cleanup intermediate artifacts after each run')
    unix_parser.add_argument('--keep_folders', type=str, help='Comma-separated folder names to keep (default: 15_test_results_after_train_csv,metrics_output,logs)')

    # 直到達到門檻模式
    until_parser = subparsers.add_parser('until', help='Run experiments until metric reaches threshold')
    until_parser.add_argument('dataset', choices=['split3', 'split4', 'apidms', 'apidms_regroup','split4_10%noise', 'split4_20%noise', 'split4_50%noise', 'apidms_10%noise', 'apidms_20%noise', 'apidms_50%noise', 'win10_split4', 'plusRed_00177', 'plusRed_00177_fulltime', 'vs0177_full_split2-1', 'ka_malBazzar'])
    until_parser.add_argument('shots', type=int, choices=[5, 10])
    until_parser.add_argument('threshold_value', type=float, help='Threshold value to reach (e.g., 0.9 for 90%)')
    until_parser.add_argument('--threshold_metric', default='accuracy', choices=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], help='Metric to check threshold')
    until_parser.add_argument('--max_runs', type=int, default=None, help='Safety cap on number of runs')
    until_parser.add_argument('--sleep_seconds', type=float, default=1.0, help='Sleep between runs to ensure different seeds')
    until_parser.add_argument('--cleanup', action='store_true', help='Cleanup intermediate artifacts after each run')
    until_parser.add_argument('--keep_folders', type=str, help='Comma-separated folder names to keep (default: 15_test_results_after_train_csv,metrics_output,logs)')
    
    # 批量實驗模式
    batch_parser = subparsers.add_parser('batch', help='Run batch experiments')
    batch_parser.add_argument('batch_name', choices=list(EXPERIMENT_BATCHES.keys()))
    batch_parser.add_argument('--cleanup', action='store_true', help='Cleanup intermediate artifacts after each run')
    batch_parser.add_argument('--keep_folders', type=str, help='Comma-separated folder names to keep (default: 15_test_results_after_train_csv,metrics_output,logs)')
    
    # 自定義批量模式
    custom_parser = subparsers.add_parser('custom', help='Run custom experiments')
    custom_parser.add_argument('experiments', help='Comma-separated experiments: "split3 5 456,apidms 10 123"')
    
    # 列出可用選項
    list_parser = subparsers.add_parser('list', help='List available options')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # 檢測運行環境
    environment = 'runpod' if '/workspace' in os.getcwd() else 'local'
    # Load hyperparameter overrides if provided (only for unix_seed for now)
    hparam_overrides = None
    if args.mode == 'unix_seed' and getattr(args, 'hparam_config', None):
        try:
            with open(args.hparam_config, 'r', encoding='utf-8') as f:
                hparam_overrides = json.load(f)
            print(f"Loaded hyperparameter overrides from {args.hparam_config}: {hparam_overrides}")
        except Exception as e:
            print(f"Warning: failed to load hparam_config: {e}")
    runner = ExperimentRunner(environment, hparam_overrides)
    
    if args.mode == 'single':
        # 處理 random_seed 參數
        if args.random_seed.lower() == 'unix':
            random_seed = 'unix'
        else:
            random_seed = int(args.random_seed)
        
        print(f"Running single experiment: {args.dataset} {args.shots}shot seed={random_seed}")
        success, result_path, actual_seed = runner.run_single_experiment(args.dataset, args.shots, random_seed)
        if success:
            print(f"Results saved to: {result_path}")
            print(f"Actual seed used: {actual_seed}")
            if getattr(args, 'cleanup', False) and result_path:
                keep = [s for s in (args.keep_folders.split(',') if getattr(args, 'keep_folders', None) else ['15_test_results_after_train_csv','metrics_output','logs']) if s]
                print("Cleaning up intermediate artifacts...")
                runner.cleanup_intermediate_artifacts(result_path, keep=keep)
        else:
            sys.exit(1)
    
    elif args.mode == 'unix_seed':
        keep = [s for s in (args.keep_folders.split(',') if getattr(args, 'keep_folders', None) else ['15_test_results_after_train_csv','metrics_output','logs']) if s]
        runner.run_unix_seed_experiments(args.dataset, args.shots, args.repeat_count, cleanup=getattr(args, 'cleanup', False), keep_folders=keep)
        # After runs, apply threshold check across aggregated CSV if requested
        if args.threshold_value is not None:
            try:
                import pandas as pd
                aggregate_dir = os.path.join(runner.config.base_paths['result_base'], 'unix_seed_summaries')
                latest_csv = None
                if os.path.isdir(aggregate_dir):
                    files = sorted([f for f in os.listdir(aggregate_dir) if f.startswith(f"summary_{args.dataset}_{args.shots}shot_") and f.endswith('.csv')])
                    if files:
                        latest_csv = os.path.join(aggregate_dir, files[-1])
                if latest_csv and os.path.exists(latest_csv):
                    df = pd.read_csv(latest_csv)
                    # Choose column
                    metric_col_map = {
                        'accuracy': 'accuracy_mean',
                        'f1_macro': 'f1_macro_mean',
                        'precision_macro': 'precision_macro_mean',
                        'recall_macro': 'recall_macro_mean',
                    }
                    col = metric_col_map.get(args.threshold_metric)
                    if col in df.columns:
                        df['meets_threshold'] = df[col] >= args.threshold_value
                        pass_count = int(df['meets_threshold'].sum())
                        total = len(df)
                        print(f"\nThreshold check on {args.threshold_metric} >= {args.threshold_value}:")
                        print(f"  Passed: {pass_count}/{total}")
                        # Save annotated CSV
                        annotated_csv = latest_csv.replace('.csv', '_with_threshold.csv')
                        df.to_csv(annotated_csv, index=False)
                        print(f"  Saved annotated summary: {annotated_csv}")
                    else:
                        print(f"Warning: metric column '{col}' not found in summary {latest_csv}")
                else:
                    print("Warning: no aggregated summary CSV found to apply threshold check.")
            except Exception as e:
                print(f"Warning: failed threshold check processing: {e}")
    
    elif args.mode == 'batch':
        keep = [s for s in (args.keep_folders.split(',') if getattr(args, 'keep_folders', None) else ['15_test_results_after_train_csv','metrics_output','logs']) if s]
        runner.run_batch_experiments(args.batch_name, cleanup=getattr(args, 'cleanup', False), keep_folders=keep)
    
    elif args.mode == 'custom':
        experiments = []
        for exp_str in args.experiments.split(','):
            parts = exp_str.strip().split()
            if len(parts) == 3:
                dataset, shots, seed_str = parts[0], int(parts[1]), parts[2]
                seed = 'unix' if seed_str.lower() == 'unix' else int(seed_str)
                experiments.append((dataset, shots, seed))
        
        print(f"Running {len(experiments)} custom experiments")
        for dataset, shots, seed in experiments:
            runner.run_single_experiment(dataset, shots, seed)
    
    elif args.mode == 'list':
        print("Available datasets: split3, split4, apidms, apidms_regroup, win10_split4, plusRed_00177, plusRed_00177_fulltime, vs0177_full_split2-1")  # 新增 plusRed_00177_fulltime
        print("Available shots: 5, 10")
        print("Random seed: any integer or 'unix' for current timestamp")
        print(f"Available batches: {list(EXPERIMENT_BATCHES.keys())}")
        print("\nExample commands:")
        print("  python run_experiments.py single split4 5 unix")  # 新增 split4 範例
        print("  python run_experiments.py unix_seed split4 5 3")  # 新增 split4 範例
        print("  python run_experiments.py batch unix_seed_test")
        print("  python run_experiments.py until split4 5 0.9 --threshold_metric accuracy --max_runs 50")

    elif args.mode == 'until':
        keep = [s for s in (args.keep_folders.split(',') if getattr(args, 'keep_folders', None) else ['15_test_results_after_train_csv','metrics_output','logs']) if s]
        success, result_path, seed, metric = runner.run_until_threshold(
            args.dataset, args.shots, args.threshold_metric, args.threshold_value,
            max_runs=args.max_runs, sleep_seconds=args.sleep_seconds, cleanup=getattr(args, 'cleanup', False), keep_folders=keep
        )
        if success:
            print(f"Success: reached {args.threshold_metric}>={args.threshold_value}. Result: {result_path}")
        else:
            print(f"Stopped without reaching threshold. Best {args.threshold_metric}={metric} -> {result_path}")

if __name__ == "__main__":
    main()