#!/usr/bin/env python3
"""
實驗配置文件
experiment_config.py
"""

import os
import time

class ExperimentConfig:
    """實驗配置類"""
    
    def __init__(self, environment='local'):
        self.environment = environment
        self._setup_paths()
        self._setup_datasets()
        self._setup_experiment_params()
    
    def _setup_paths(self):
        """設置基礎路徑"""
        if self.environment == 'runpod':
            self.base_paths = {
                'training_base': "/workspace/data_with_APIDMS_img/training",
                'testing_base': "/workspace/data_with_APIDMS_img/testing", 
                'result_base': "/workspace/results",
                'template_dir': "/workspace/Results_folder_structure_2"
            }
        else:  # local Windows
            self.base_paths = {
                'training_base': "C:/Users/NT/Documents/Mycode/Lizz_final_code/final_code/data/training",
                'testing_base': "C:/Users/NT/Documents/Mycode/Lizz_final_code/final_code/data/testing",
                'result_base': "C:/Users/NT/Documents/Mycode/Lizz_final_code/final_code_clean/results",
                'template_dir': "Results_folder_structure_2"
            }
    
    def _setup_datasets(self):
        """設置資料集配置"""
        self.dataset_configs = {
            'split3': {
                'training_data_suffix': 'split3_train_regrouped_for_auto',
                'training_cluster_suffix': 'split3_train_regrouped_copy',
                'testing_data_template': '9_prepare_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'split4': {
                'training_data_suffix': 'split4_train_regrouped_for_auto',
                'training_cluster_suffix': 'split4_train_regrouped_copy',
                'testing_data_template': 'split4_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'apidms': {
                'training_data_suffix': 'a_original_for_auto', 
                'training_cluster_suffix': 'a_original_copy',
                'testing_data_template': 'A_prepare_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'apidms_regroup': {
                'training_data_suffix': 'APIDMS_img_train_regrouped_for_auto',
                'training_cluster_suffix': 'APIDMS_img_train_regrouped_copy',
                'testing_data_template': 'APIDMS_img_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'split4_10%noise': {
                'training_data_suffix': 'split4_train_regrouped_10percent_noise',
                'training_cluster_suffix': 'split4_train_regrouped_copy_10percent_noise',
                'testing_data_template': 'split4_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'split4_20%noise': {
                'training_data_suffix': 'split4_train_regrouped_20percent_noise',
                'training_cluster_suffix': 'split4_train_regrouped_copy_20percent_noise',
                'testing_data_template': 'split4_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'split4_50%noise': {
                'training_data_suffix': 'split4_train_regrouped_50percent_noise',
                'training_cluster_suffix': 'split4_train_regrouped_copy_50percent_noise',
                'testing_data_template': 'split4_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'apidms_10%noise': {
                'training_data_suffix': 'APIDMS_img_train_regrouped_10percent_noise',
                'training_cluster_suffix': 'APIDMS_img_train_regrouped_copy_10percent_noise',
                'testing_data_template': 'A_prepare_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'apidms_20%noise': {
                'training_data_suffix': 'APIDMS_img_train_regrouped_20percent_noise',
                'training_cluster_suffix': 'APIDMS_img_train_regrouped_copy_20percent_noise',
                'testing_data_template': 'A_prepare_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'apidms_50%noise': {
                'training_data_suffix': 'APIDMS_img_train_regrouped_50percent_noise',
                'training_cluster_suffix': 'APIDMS_img_train_regrouped_copy_50percent_noise',
                'testing_data_template': 'A_prepare_data_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'win10_split4': {
                'training_data_suffix': 'split4_train_regrouped_for_auto',
                'training_cluster_suffix': 'split4_train_regrouped_copy',
                'testing_data_template': 'win10_120s_split4_{shots}shots_5_10_35cases'
            },
            'plusRed_00177': {
                'training_data_suffix': 'plusRed_00177_train_regrouped_for_auto',
                'training_cluster_suffix': 'plusRed_00177_train_regrouped_copy',
                'testing_data_template': 'balanced_split3_{shots}shots_5_10_35cases'
            },
            'plusRed_00177_fulltime': {
                'training_data_suffix': 'plusRed_00177_fulltime_train_regrouped_for_auto',
                'training_cluster_suffix': 'plusRed_00177_fulltime_train_regrouped_copy',
                'testing_data_template': 'plusRed_00177_fulltime_{shots}shots_5_10_35cases'
            },
            'vs0177_full_split2-1': {
                'training_data_suffix': 'vs0177_full_split2-1_train_regrouped_for_auto',
                'training_cluster_suffix': 'vs0177_full_split2-1_train_regrouped_copy',
                'testing_data_template': 'vs0177_full_split2-1_for_method_1_split3_{shots}shots_5_10_35cases'
            },
            'ka_malBazzar': {
                'training_data_suffix': 'ka_malBazzar_train_regrouped_for_auto',
                'training_cluster_suffix': 'ka_malBazzar_train_regrouped_copy',
                'testing_data_template': 'ka_malBazzar_for_method_1_split3_{shots}shots_5_10_35cases'
            },
        }
    
    def _setup_experiment_params(self):
        """設置實驗參數"""
        self.experiment_params = {
            'si_epoch_num': 65,
            'way_list': [5, 10],
            'net_name': "SiameseNetwork_autoencoder_Based",
            'net_type': "vgg",
            'case_len': 35,
            'model_num': 2,
            'auto_epoch_num': 150,
            # 驗證頻率（每多少個 epoch 做一次驗證）
            'validate_every_epochs': 5,
            # 控制是否輸出每個 case 的 PR-AUC 圖片
            'save_pr_curve_images': False,
            # Early stopping defaults
            'early_stopping_enable': False,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 0.0,
            'early_stopping_monitor': 'accuracy'
        }
    
    def get_dataset_paths(self, dataset_name, shots):
        """獲取資料集路徑"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        testing_dir = config['testing_data_template'].format(shots=shots)
        
        return {
            'autoencoder_training_data_path': os.path.join(
                self.base_paths['training_base'], 
                config['training_data_suffix']
            ),
            'training_data_cluster_target': os.path.join(
                self.base_paths['training_base'], 
                config['training_cluster_suffix']
            ),
            'testing_data_dir': os.path.join(
                self.base_paths['testing_base'], 
                testing_dir, 
                'unknown_with_case'
            ),
            'proto_data_dir': os.path.join(
                self.base_paths['testing_base'], 
                testing_dir, 
                'known_with_case'
            )
        }
    
    def get_result_folder_name(self, dataset_name, shots, random_seed=None, run_id=None):
        """生成結果資料夾名稱"""
        if run_id is not None:
            return f"R_{dataset_name}_{shots}s_r{run_id:03d}"
        else:
            return f"R_{dataset_name}_{shots}s_s{random_seed}"
    
    def validate_paths(self, dataset_paths):
        """驗證路徑是否存在"""
        missing_paths = []
        for name, path in dataset_paths.items():
            if not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("Warning: The following paths do not exist:")
            for path in missing_paths:
                print(f"  - {path}")
        
        return len(missing_paths) == 0
    
    @staticmethod
    def generate_unix_timestamp_seed():
        """生成基於當前 Unix 時間戳的隨機種子"""
        return int(time.time())

# 預定義的實驗批次
EXPERIMENT_BATCHES = {
    'quick_test': [
        ('split3', 5, 456),
    ],
    
    'quick_test_split4': [
        ('split4', 5, 456),
    ],
    
    'quick_test_apidms_regroup': [
        ('apidms_regroup', 5, 456),
    ],
    
    'full_split3': [
        ('split3', 5, 2),
        ('split3', 5, 3),
        ('split3', 5, 4),
        ('split3', 5, 5),
        ('split3', 5, 6),
        ('split3', 5, 7),
        ('split3', 5, 8),
        ('split3', 5, 9),
        ('split3', 5, 10),
        ('split3', 5, 11),
        ('split3', 5, 12),
        ('split3', 5, 13),
        ('split3', 5, 14),
        ('split3', 5, 15),
        ('split3', 10, 2),
        ('split3', 10, 3),
        ('split3', 10, 4),
        ('split3', 10, 5),
        ('split3', 10, 6),
        ('split3', 10, 7),
        ('split3', 10, 8),
        ('split3', 10, 9),
        ('split3', 10, 10),
        ('split3', 10, 11),
        ('split3', 10, 12),
        ('split3', 10, 13),
        ('split3', 10, 14),
        ('split3', 10, 15),
    ],
    
    'full_split4': [
        ('split4', 5, 2),
        ('split4', 5, 3),
        ('split4', 5, 4),
        ('split4', 5, 5),
        ('split4', 5, 6),
        ('split4', 5, 7),
        ('split4', 5, 8),
        ('split4', 5, 9),
        ('split4', 5, 10),
        ('split4', 5, 11),
        ('split4', 5, 12),
        ('split4', 5, 13),
        ('split4', 5, 14),
        ('split4', 5, 15),
        ('split4', 10, 2),
        ('split4', 10, 3),
        ('split4', 10, 4),
        ('split4', 10, 5),
        ('split4', 10, 6),
        ('split4', 10, 7),
        ('split4', 10, 8),
        ('split4', 10, 9),
        ('split4', 10, 10),
        ('split4', 10, 11),
        ('split4', 10, 12),
        ('split4', 10, 13),
        ('split4', 10, 14),
        ('split4', 10, 15),
    ],
    
    'full_apidms': [
        ('apidms', 5, 2),
        ('apidms', 5, 3),
        ('apidms', 5, 4),
        ('apidms', 5, 5),
        ('apidms', 5, 6),
        ('apidms', 5, 7),
        ('apidms', 5, 8),
        ('apidms', 5, 9),
        ('apidms', 5, 10),
        ('apidms', 5, 11),
        ('apidms', 5, 12),
        ('apidms', 5, 13),
        ('apidms', 5, 14),
        ('apidms', 5, 15),
        ('apidms', 10, 2),
        ('apidms', 10, 3),
        ('apidms', 10, 4),
        ('apidms', 10, 5),
        ('apidms', 10, 6),
        ('apidms', 10, 7),
        ('apidms', 10, 8),
        ('apidms', 10, 9),
        ('apidms', 10, 10),
        ('apidms', 10, 11),
        ('apidms', 10, 12),
        ('apidms', 10, 13),
        ('apidms', 10, 14),
        ('apidms', 10, 15),
    ],
    
    'full_apidms_regroup': [
        ('apidms_regroup', 5, 2),
        ('apidms_regroup', 5, 3),
        ('apidms_regroup', 5, 4),
        ('apidms_regroup', 5, 5),
        ('apidms_regroup', 5, 6),
        ('apidms_regroup', 5, 7),
        ('apidms_regroup', 5, 8),
        ('apidms_regroup', 5, 9),
        ('apidms_regroup', 5, 10),
        ('apidms_regroup', 5, 11),
        ('apidms_regroup', 5, 12),
        ('apidms_regroup', 5, 13),
        ('apidms_regroup', 5, 14),
        ('apidms_regroup', 5, 15),
        ('apidms_regroup', 10, 2),
        ('apidms_regroup', 10, 3),
        ('apidms_regroup', 10, 4),
        ('apidms_regroup', 10, 5),
        ('apidms_regroup', 10, 6),
        ('apidms_regroup', 10, 7),
        ('apidms_regroup', 10, 8),
        ('apidms_regroup', 10, 9),
        ('apidms_regroup', 10, 10),
        ('apidms_regroup', 10, 11),
        ('apidms_regroup', 10, 12),
        ('apidms_regroup', 10, 13),
        ('apidms_regroup', 10, 14),
        ('apidms_regroup', 10, 15),
    ],
    
    'all_experiments': [
        ('split3', 5, 456), ('split3', 5, 123), ('split3', 5, 789),
        ('split3', 10, 456), ('split3', 10, 123), ('split3', 10, 789),
        ('split4', 5, 456), ('split4', 5, 123), ('split4', 5, 789),
        ('split4', 10, 456), ('split4', 10, 123), ('split4', 10, 789),
        ('apidms', 5, 456), ('apidms', 5, 123), ('apidms', 5, 789),
        ('apidms', 10, 456), ('apidms', 10, 123), ('apidms', 10, 789),
        ('apidms_regroup', 5, 456), ('apidms_regroup', 5, 123), ('apidms_regroup', 5, 789),
        ('apidms_regroup', 10, 456), ('apidms_regroup', 10, 123), ('apidms_regroup', 10, 789),
    ],
    
    # 新增：使用當前時間種子的實驗批次
    'unix_seed_test': [
        # 格式: (dataset, shots, 'unix', repeat_count)
        ('split3', 5, 'unix', 3),
        ('split3', 10, 'unix', 3),
        ('split4', 5, 'unix', 3),
        ('split4', 10, 'unix', 3),
        ('apidms_regroup', 5, 'unix', 3),
        ('apidms_regroup', 10, 'unix', 3),
    ],
    
    'unix_seed_full': [
        ('split3', 5, 'unix', 5),
        ('split3', 10, 'unix', 5),
        ('split4', 5, 'unix', 5),
        ('split4', 10, 'unix', 5),
        ('apidms', 5, 'unix', 5),
        ('apidms', 10, 'unix', 5),
        ('apidms_regroup', 5, 'unix', 5),
        ('apidms_regroup', 10, 'unix', 5),
    ],

    'quick_test_ka_malBazzar': [
        ('ka_malBazzar', 5, 456),
    ],

    'full_ka_malBazzar': [
        ('ka_malBazzar', 5, 'unix', 5),
        ('ka_malBazzar', 10, 'unix', 5),
    ],
}