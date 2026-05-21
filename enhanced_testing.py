"""
Enhanced Testing Functions for Confusion Matrix and PR-AUC Calculation
This module provides enhanced versions of testing functions that can handle
multiple test cases and calculate detailed metrics including confusion matrix and PR-AUC.
"""

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

# Import your existing modules
from dataset import SiameseNetworkDataset_1, PrototypeDataset
from utils import Testing_Loop_0



def group_test_with_metrics(model_num=1, way_list=[5, 10], case_len=10, epoch=0, model_path="", batch_num=0,
               net_name="", net_type="",
               test_data_dir="", test_data_postfix="",
               proto_data_dir="", proto_data_postfix="", save_metrics_dir="", random_seed=42, encoder_path=None,
               save_pr_curve_images=True, shots=None):
    """
    Enhanced group_test function with comprehensive PR-AUC curve storage for each case.
    """
    
    # 設置random seed確保可重現性
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        print(f"Random seed set to: {random_seed}")
    
    WAYLIST = way_list
    CASELEN = case_len
    NET_TYPE = net_type
    result_df = pd.DataFrame()
    avg_df = pd.DataFrame()
    all_metrics = {}
    # 追加：建立與 result/avg 結構相同的 PRF 輸出表
    precision_result_df = pd.DataFrame()
    recall_result_df = pd.DataFrame()
    f1_result_df = pd.DataFrame()
    precision_avg_df = pd.DataFrame()
    recall_avg_df = pd.DataFrame()
    f1_avg_df = pd.DataFrame()
    
    # Create save directory structure
    if save_metrics_dir and not os.path.exists(save_metrics_dir):
        os.makedirs(save_metrics_dir)
    
    # Create subdirectories for organized storage
    pr_curves_dir = os.path.join(save_metrics_dir, "pr_curves")
    pr_data_dir = os.path.join(save_metrics_dir, "pr_data")
    confusion_matrices_dir = os.path.join(save_metrics_dir, "confusion_matrices")
    
    for subdir in [pr_curves_dir, pr_data_dir, confusion_matrices_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
    # Storage for all PR-AUC results
    all_pr_results = []
    
    for j in range(len(WAYLIST)):
        WAYS = WAYLIST[j]
        cases_result = []
        cases_predictions = []
        cases_true_labels = []
        
        # 為該way數創建子目錄
        way_pr_curves_dir = os.path.join(pr_curves_dir, f"{WAYS}ways")
        way_pr_data_dir = os.path.join(pr_data_dir, f"{WAYS}ways")
        
        for subdir in [way_pr_curves_dir, way_pr_data_dir]:
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        
        way_case_results = []
        
        for k in range(CASELEN):
            CASE = k+1
            print(f'{WAYS} WAYS CASE {CASE}')
            TESTINGDATAPATH = os.sep.join([os.path.normpath(test_data_dir),f"test_{WAYS}_{CASE}{test_data_postfix}"])
            PROTODATAPATH =  os.sep.join([os.path.normpath(proto_data_dir),f"test_{WAYS}_{CASE}{proto_data_postfix}"])
            print("TESTINGDATAPATH: ", TESTINGDATAPATH)
            print("PROTODATAPATH: ", PROTODATAPATH)
            MODELPATH = model_path
            EPOCH = epoch

            # Resize the images and transform to tensors
            # transformation = transforms.Compose([transforms.Resize((64,64)),
            transformation = transforms.Compose([transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
                                                 transforms.ToTensor()
                                                ])

            # Locate the test dataset and load it into the SiameseNetworkDataset
            folder_dataset_test = datasets.ImageFolder(root=TESTINGDATAPATH)
            siamese_test_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset_test,
                                                    transform=transformation)

            # Locate the test dataset and load it into the SiameseNetworkDataset
            if model_num != 1:
                pr_data_path = PROTODATAPATH
                pr_folder_dataset_test = datasets.ImageFolder(root=PROTODATAPATH)
                pr_dataset = PrototypeDataset(imageFolderDataset=pr_folder_dataset_test,
                                                        transform=transformation)
            else:
                pr_data_path = ""
                pr_dataset = ""

            # Use enhanced testing loop that returns predictions and labels
            result_dict = Testing_Loop_with_predictions(model_num=model_num, dataset=siamese_test_dataset, 
                         net_name=net_name, model_path=MODELPATH,
                         testing_data_path=TESTINGDATAPATH, pr_dataset=pr_dataset, pr_data_path=pr_data_path, encoder_path=encoder_path)
            
            # 處理該案例的結果
            case_predictions = result_dict["predictions"]
            case_true_labels = result_dict["true_labels"]
            case_similarities = result_dict["similarities"]
            case_unique_classes = sorted(list(set(case_true_labels)))
            
            print(f"Case {CASE}: found {len(case_unique_classes)} unique classes: {case_unique_classes}")
            
            # 計算並保存該案例的PR-AUC
            case_pr_data = None
            if (len(case_unique_classes) == WAYS and 
                isinstance(case_similarities, np.ndarray) and 
                len(case_similarities.shape) == 2 and 
                case_similarities.shape[1] == WAYS):
                
                try:
                    # 計算PR-AUC並保存詳細數據
                    case_pr_data = calculate_and_save_case_pr_auc(
                        case_true_labels, 
                        case_similarities, 
                        case_unique_classes,
                        WAYS, CASE, 
                        way_pr_curves_dir, 
                        way_pr_data_dir,
                        save_image=save_pr_curve_images
                    )
                    print(f"Case {CASE} Average PR-AUC: {case_pr_data['avg_pr_auc']:.4f}")
                    
                except Exception as e:
                    print(f"Error processing Case {CASE}: {e}")
            
            # 追加：計算該案例的 precision/recall/F1（macro 平均）
            case_prf_report = None
            case_macro_precision = None
            case_macro_recall = None
            case_macro_f1 = None
            try:
                if len(case_predictions) > 0 and len(set(case_true_labels)) > 1:
                    case_prf_report = classification_report(
                        case_true_labels,
                        case_predictions,
                        labels=case_unique_classes,
                        output_dict=True,
                        zero_division=0
                    )
                    # 使用 macro avg 作為每個案例的代表指標
                    if 'macro avg' in case_prf_report:
                        case_macro_precision = case_prf_report['macro avg'].get('precision', 0.0)
                        case_macro_recall = case_prf_report['macro avg'].get('recall', 0.0)
                        case_macro_f1 = case_prf_report['macro avg'].get('f1-score', 0.0)
                        print(f"Case {CASE} Macro - Precision: {case_macro_precision:.4f}, Recall: {case_macro_recall:.4f}, F1: {case_macro_f1:.4f}")
            except Exception as e:
                print(f"Warning: failed to compute per-case PRF for Case {CASE}: {e}")
            
            # 儲存案例結果
            case_result = {
                'ways': WAYS,
                'case_id': CASE,
                'accuracy': result_dict["accuracy"],
                'predictions': case_predictions,
                'true_labels': case_true_labels,
                'similarities': case_similarities,
                'unique_classes': case_unique_classes,
                'pr_data': case_pr_data,
                # 追加：每案 PRF 指標
                'macro_precision': case_macro_precision,
                'macro_recall': case_macro_recall,
                'macro_f1': case_macro_f1,
                'classification_report': case_prf_report
            }
            way_case_results.append(case_result)
            all_pr_results.append(case_result)
            
            cases_result.append(result_dict["accuracy"])
            cases_predictions.extend(case_predictions)
            cases_true_labels.extend(case_true_labels)
        
        # 計算該way數的統計（accuracy）
        col_name = f'{EPOCH}EPOCH{batch_num}BATCH-{WAYS}WAYS'
        result_df[col_name] = cases_result
        avg_df[col_name] = [np.average(cases_result)]

        # 追加：填入 PRF（macro）對應的 result/avg 表
        precisions_series = [c['macro_precision'] if c['macro_precision'] is not None else np.nan for c in way_case_results]
        recalls_series = [c['macro_recall'] if c['macro_recall'] is not None else np.nan for c in way_case_results]
        f1s_series = [c['macro_f1'] if c['macro_f1'] is not None else np.nan for c in way_case_results]

        precision_result_df[col_name] = precisions_series
        recall_result_df[col_name] = recalls_series
        f1_result_df[col_name] = f1s_series

        precision_avg_df[col_name] = [np.nanmean(precisions_series) if len(precisions_series) > 0 else np.nan]
        recall_avg_df[col_name] = [np.nanmean(recalls_series) if len(recalls_series) > 0 else np.nan]
        f1_avg_df[col_name] = [np.nanmean(f1s_series) if len(f1s_series) > 0 else np.nan]
        
        # 提取有效的PR-AUC結果
        valid_pr_aucs = [case['pr_data']['avg_pr_auc'] for case in way_case_results 
                        if case['pr_data'] is not None]
        
        if valid_pr_aucs:
            print(f"\n{WAYS}-way PR-AUC Statistics:")
            print(f"  Mean: {np.mean(valid_pr_aucs):.4f}")
            print(f"  Std:  {np.std(valid_pr_aucs):.4f}")
            print(f"  Valid cases: {len(valid_pr_aucs)}/{len(way_case_results)}")
        
        # 追加：提取有效的 per-case PRF（macro）並計算平均
        valid_macro_precisions = [c['macro_precision'] for c in way_case_results if c['macro_precision'] is not None]
        valid_macro_recalls = [c['macro_recall'] for c in way_case_results if c['macro_recall'] is not None]
        valid_macro_f1s = [c['macro_f1'] for c in way_case_results if c['macro_f1'] is not None]
        avg_macro_precision = np.mean(valid_macro_precisions) if valid_macro_precisions else None
        avg_macro_recall = np.mean(valid_macro_recalls) if valid_macro_recalls else None
        avg_macro_f1 = np.mean(valid_macro_f1s) if valid_macro_f1s else None
        if avg_macro_precision is not None and avg_macro_recall is not None and avg_macro_f1 is not None:
            print(f"{WAYS}-way Macro Averages - Precision: {avg_macro_precision:.4f}, Recall: {avg_macro_recall:.4f}, F1: {avg_macro_f1:.4f}")
        
        # 基礎的 way 級別指標（無論是否能計算混淆矩陣都要提供，避免 KeyError）
        way_unique_classes = sorted(list(set(cases_true_labels))) if len(cases_true_labels) > 0 else []
        way_metrics = {
            'avg_pr_auc': np.mean(valid_pr_aucs) if valid_pr_aucs else None,
            'pr_auc_per_class': [],
            'confusion_matrix': None,
            'case_level_pr_aucs': valid_pr_aucs,
            'case_level_avg_pr_auc': np.mean(valid_pr_aucs) if valid_pr_aucs else 0.0,
            'classification_report': None,
            'unique_classes': way_unique_classes,
            'valid_cases_count': len(valid_pr_aucs),
            'total_cases_count': len(way_case_results),
            # 追加：way 級別 PRF（macro 平均）
            'avg_precision': avg_macro_precision,
            'avg_recall': avg_macro_recall,
            'avg_f1': avg_macro_f1
        }
        
        # 計算混淆矩陣（若條件允許則補上）
        if len(cases_predictions) > 0 and len(set(cases_true_labels)) > 1:
            cm = confusion_matrix(cases_true_labels, cases_predictions, labels=way_unique_classes)
            
            # 保存混淆矩陣
            cm_save_path = os.path.join(confusion_matrices_dir, f'confusion_matrix_{WAYS}ways.png')
            save_confusion_matrix_plot(cm, f'{WAYS} Ways', cm_save_path, way_unique_classes)
            
            # 更新 way 指標中的矩陣與分類報告
            way_metrics.update({
                'confusion_matrix': cm,
                'classification_report': classification_report(cases_true_labels, cases_predictions, output_dict=True)
            })
        
        # 儲存 way 級別的指標（含預設鍵）
        all_metrics[f'{WAYS}_ways'] = way_metrics
        
        # 追加：輸出 per-case PRF 指標到 CSV
        try:
            per_case_rows = []
            for c in way_case_results:
                per_case_rows.append({
                    'ways': c['ways'],
                    'case_id': c['case_id'],
                    'accuracy': c['accuracy'],
                    'macro_precision': c['macro_precision'],
                    'macro_recall': c['macro_recall'],
                    'macro_f1': c['macro_f1'],
                    'avg_pr_auc': c['pr_data']['avg_pr_auc'] if c['pr_data'] is not None else None
                })
            per_case_df = pd.DataFrame(per_case_rows)
            per_case_csv_path = os.path.join(way_pr_data_dir, f'{WAYS}ways_per_case_metrics.csv')
            per_case_df.to_csv(per_case_csv_path, index=False)
            print(f"Saved per-case PRF metrics: {per_case_csv_path}")
        except Exception as e:
            print(f"Warning: failed to save per-case PRF CSV for {WAYS} ways: {e}")
        
        # 創建該way數的PR-AUC總結報告
        create_way_pr_summary(way_case_results, WAYS, way_pr_data_dir)
    
    # 創建全局PR-AUC總結報告
    create_global_pr_summary(all_pr_results, pr_data_dir)
    
    # 計算整體混淆矩陣（僅用於參考）
    all_predictions = []
    all_true_labels = []
    for case_result in all_pr_results:
        all_predictions.extend(case_result['predictions'])
        all_true_labels.extend(case_result['true_labels'])
    
    if len(all_predictions) > 0:
        overall_unique_classes = sorted(list(set(all_true_labels)))
        overall_cm = confusion_matrix(all_true_labels, all_predictions)
        
        cm_save_path = os.path.join(confusion_matrices_dir, 'confusion_matrix_overall.png')
        save_confusion_matrix_plot(overall_cm, 'Overall - All Ways Combined', cm_save_path, overall_unique_classes)
        
        overall_metrics = {
            'confusion_matrix': overall_cm,
            'pr_auc_per_class': [],  # 不計算，因為類別混合
            'avg_pr_auc': 0.0,       # 不計算，因為類別不一致
            'case_level_pr_aucs': [],
            'case_level_avg_pr_auc': 0.0,
            'classification_report': classification_report(all_true_labels, all_predictions, output_dict=True),
            'unique_classes': overall_unique_classes,
            'note': 'Overall metrics for reference only - PR-AUC not calculated due to mixed class sets'
        }
        all_metrics['overall'] = overall_metrics
    
    print("\n" + "="*50)
    print("ACCURACY RESULTS:")
    print(result_df)
    print("\nAVERAGE ACCURACY:")
    print(avg_df)
    
    # 追加：輸出與 accuracy 相同結構的 PRF 檔案
    try:
        if save_metrics_dir:
            precision_result_csv = os.path.join(save_metrics_dir, 'precision_result.csv')
            precision_avg_csv = os.path.join(save_metrics_dir, 'precision_avg.csv')
            recall_result_csv = os.path.join(save_metrics_dir, 'recall_result.csv')
            recall_avg_csv = os.path.join(save_metrics_dir, 'recall_avg.csv')
            f1_result_csv = os.path.join(save_metrics_dir, 'f1_result.csv')
            f1_avg_csv = os.path.join(save_metrics_dir, 'f1_avg.csv')
            precision_result_df.to_csv(precision_result_csv, index=False)
            precision_avg_df.to_csv(precision_avg_csv, index=False)
            recall_result_df.to_csv(recall_result_csv, index=False)
            recall_avg_df.to_csv(recall_avg_csv, index=False)
            f1_result_df.to_csv(f1_result_csv, index=False)
            f1_avg_df.to_csv(f1_avg_csv, index=False)
            print("Saved PRF CSVs:")
            print(f"  {precision_result_csv}")
            print(f"  {precision_avg_csv}")
            print(f"  {recall_result_csv}")
            print(f"  {recall_avg_csv}")
            print(f"  {f1_result_csv}")
            print(f"  {f1_avg_csv}")

            # 也輸出 PR-AUC 的總覽
            try:
                pr_auc_csv = os.path.join(save_metrics_dir, 'pr_auc_overview.csv')
                rows = []
                for way_key, m in all_metrics.items():
                    if way_key == 'overall':
                        continue
                    rows.append({
                        'way': way_key,
                        'avg_pr_auc': m.get('avg_pr_auc', None),
                        'valid_cases_count': m.get('valid_cases_count', 0)
                    })
                if rows:
                    pd.DataFrame(rows).to_csv(pr_auc_csv, index=False)
                    print(f"  {pr_auc_csv}")
            except Exception:
                pass

            # 彙整 per-way 平均並輸出 way_summary.csv 到兩個目錄
            try:
                def extract_avg_from(df: pd.DataFrame, ways: int) -> float:
                    if df is None or df.empty:
                        return float('nan')
                    cols = [c for c in df.columns if c.endswith(f'-{ways}WAYS')]
                    if not cols:
                        return float('nan')
                    try:
                        return float(pd.to_numeric(df[cols[0]].iloc[0], errors='coerce'))
                    except Exception:
                        return float('nan')

                summary_rows = []
                for ways in way_list:
                    summary_rows.append({
                        'seed': random_seed,
                        'shots': shots,
                        'way': ways,
                        'accuracy_mean': extract_avg_from(avg_df, ways),
                        'precision_macro_mean': extract_avg_from(precision_avg_df, ways),
                        'recall_macro_mean': extract_avg_from(recall_avg_df, ways),
                        'f1_macro_mean': extract_avg_from(f1_avg_df, ways),
                        'avg_pr_auc': (all_metrics.get(f'{ways}_ways') or {}).get('avg_pr_auc', None)
                    })
                way_summary_df = pd.DataFrame(summary_rows)
                # 15_test_results_after_train_csv
                way_summary_csv_1 = os.path.join(save_metrics_dir, 'way_summary.csv')
                way_summary_df.to_csv(way_summary_csv_1, index=False)
                print(f"Saved per-way summary: {way_summary_csv_1}")
                # metrics_output 鏡像
                base_dir = os.path.dirname(os.path.normpath(save_metrics_dir))
                metrics_output_dir = os.path.join(base_dir, 'metrics_output')
                os.makedirs(metrics_output_dir, exist_ok=True)
                way_summary_csv_2 = os.path.join(metrics_output_dir, 'way_summary.csv')
                way_summary_df.to_csv(way_summary_csv_2, index=False)
                print(f"Saved per-way summary mirror: {way_summary_csv_2}")
            except Exception as e2:
                print(f"Warning: failed to save per-way summary CSVs: {e2}")
    except Exception as e:
        print(f"Warning: failed to save PRF result/avg CSVs: {e}")

    return result_df, avg_df, all_metrics

def calculate_and_save_case_pr_auc(true_labels, similarities, unique_classes, ways, case_id, 
                                  curves_dir, data_dir, save_image=True):
    """
    計算並保存單個案例的PR-AUC曲線和數據
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # 驗證輸入
    y_true = np.array(true_labels)
    y_score = np.array(similarities)
    
    if y_score.shape[1] != len(unique_classes):
        raise ValueError(f"Similarity dimensions mismatch: {y_score.shape[1]} vs {len(unique_classes)}")
    
    # 轉換為二進制標籤
    y_true_binary = label_binarize(y_true, classes=unique_classes)
    
    # 儲存PR曲線數據
    pr_curves_data = {}
    pr_aucs = []
    
    # 創建圖表
    if save_image:
        plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(unique_classes):
        binary_true = y_true_binary[:, i]
        class_scores = y_score[:, i]
        
        positive_samples = np.sum(binary_true)
        negative_samples = len(binary_true) - positive_samples
        
        if positive_samples > 0 and negative_samples > 0:
            precision, recall, thresholds = precision_recall_curve(binary_true, class_scores)
            ap = average_precision_score(binary_true, class_scores)
            
            # 繪製曲線（可關閉）
            if save_image:
                plt.plot(recall, precision, label=f'Class {class_name} (AP={ap:.3f})')
            
            # 儲存數據
            pr_curves_data[class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(), 
                'thresholds': thresholds.tolist(),
                'average_precision': ap,
                'positive_samples': int(positive_samples),
                'negative_samples': int(negative_samples)
            }
            pr_aucs.append(ap)
        else:
            pr_curves_data[class_name] = {
                'precision': [],
                'recall': [],
                'thresholds': [],
                'average_precision': 0.0,
                'positive_samples': int(positive_samples),
                'negative_samples': int(negative_samples),
                'note': 'Skipped - insufficient samples'
            }
            pr_aucs.append(0.0)
    
    # 完成並保存圖表（可關閉）
    curve_save_path = None
    if save_image:
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curves - {ways} Ways Case {case_id}')
        plt.legend()
        plt.grid(True)
        curve_save_path = os.path.join(curves_dir, f'case_{case_id}_pr_curves.png')
        plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 計算平均PR-AUC
    avg_pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0
    
    # 準備完整的案例數據
    case_pr_data = {
        'ways': ways,
        'case_id': case_id,
        'unique_classes': unique_classes,
        'pr_curves_data': pr_curves_data,
        'pr_aucs': pr_aucs,
        'avg_pr_auc': avg_pr_auc,
        'total_samples': len(true_labels),
        'curve_image_path': curve_save_path
    }
    
    # 保存詳細數據到JSON文件
    import json
    data_save_path = os.path.join(data_dir, f'case_{case_id}_pr_data.json')
    with open(data_save_path, 'w', encoding='utf-8') as f:
        json.dump(case_pr_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved PR curves: {curve_save_path}")
    print(f"Saved PR data: {data_save_path}")
    
    return case_pr_data

def create_way_pr_summary(way_case_results, ways, data_dir):
    """
    創建特定way數的PR-AUC總結報告
    """
    import json
    
    valid_cases = [case for case in way_case_results if case['pr_data'] is not None]
    
    if not valid_cases:
        return
    
    # 統計數據
    all_pr_aucs = [case['pr_data']['avg_pr_auc'] for case in valid_cases]
    
    # 按類別統計
    class_pr_stats = {}
    for case in valid_cases:
        pr_data = case['pr_data']
        for class_name, class_data in pr_data['pr_curves_data'].items():
            if class_name not in class_pr_stats:
                class_pr_stats[class_name] = []
            if 'average_precision' in class_data:
                class_pr_stats[class_name].append(class_data['average_precision'])
    
    # 計算每個類別的統計
    class_statistics = {}
    for class_name, pr_values in class_pr_stats.items():
        if pr_values:
            class_statistics[class_name] = {
                'mean_pr_auc': np.mean(pr_values),
                'std_pr_auc': np.std(pr_values),
                'min_pr_auc': np.min(pr_values),
                'max_pr_auc': np.max(pr_values),
                'count': len(pr_values)
            }
    
    # 創建總結報告
    summary_report = {
        'ways': ways,
        'total_cases': len(way_case_results),
        'valid_cases': len(valid_cases),
        'overall_statistics': {
            'mean_avg_pr_auc': np.mean(all_pr_aucs),
            'std_avg_pr_auc': np.std(all_pr_aucs),
            'min_avg_pr_auc': np.min(all_pr_aucs),
            'max_avg_pr_auc': np.max(all_pr_aucs)
        },
        'per_class_statistics': class_statistics,
        'individual_case_results': [
            {
                'case_id': case['case_id'],
                'avg_pr_auc': case['pr_data']['avg_pr_auc'],
                'classes': case['unique_classes'],
                'individual_pr_aucs': case['pr_data']['pr_aucs']
            } for case in valid_cases
        ]
    }
    
    # 保存總結報告
    summary_path = os.path.join(data_dir, f'{ways}ways_pr_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    # 創建可讀的文本報告
    txt_path = os.path.join(data_dir, f'{ways}ways_pr_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"PR-AUC Summary Report - {ways} Ways\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Cases: {len(way_case_results)}\n")
        f.write(f"Valid Cases: {len(valid_cases)}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Mean PR-AUC: {summary_report['overall_statistics']['mean_avg_pr_auc']:.4f}\n")
        f.write(f"  Std PR-AUC:  {summary_report['overall_statistics']['std_avg_pr_auc']:.4f}\n")
        f.write(f"  Min PR-AUC:  {summary_report['overall_statistics']['min_avg_pr_auc']:.4f}\n")
        f.write(f"  Max PR-AUC:  {summary_report['overall_statistics']['max_avg_pr_auc']:.4f}\n\n")
        
        f.write("Per-Class Statistics:\n")
        for class_name, stats in class_statistics.items():
            f.write(f"  {class_name}:\n")
            f.write(f"    Mean: {stats['mean_pr_auc']:.4f}\n")
            f.write(f"    Std:  {stats['std_pr_auc']:.4f}\n")
            f.write(f"    Count: {stats['count']}\n\n")
        
        f.write("Individual Case Results:\n")
        for case_data in summary_report['individual_case_results']:
            f.write(f"  Case {case_data['case_id']}: {case_data['avg_pr_auc']:.4f}\n")
            f.write(f"    Classes: {case_data['classes']}\n")
            f.write(f"    Individual PR-AUCs: {[f'{x:.3f}' for x in case_data['individual_pr_aucs']]}\n\n")
    
    print(f"Saved {ways}-way summary: {summary_path}")
    print(f"Saved {ways}-way text report: {txt_path}")


def create_global_pr_summary(all_pr_results, data_dir):
    """
    創建跨所有way數的全局PR-AUC總結報告
    """
    import json
    
    valid_results = [result for result in all_pr_results if result['pr_data'] is not None]
    
    if not valid_results:
        return
    
    # 按way數分組
    by_ways = {}
    for result in valid_results:
        ways = result['ways']
        if ways not in by_ways:
            by_ways[ways] = []
        by_ways[ways].append(result)
    
    # 創建全局統計
    global_summary = {
        'total_cases': len(all_pr_results),
        'valid_cases': len(valid_results),
        'by_ways': {}
    }
    
    for ways, results in by_ways.items():
        pr_aucs = [result['pr_data']['avg_pr_auc'] for result in results]
        global_summary['by_ways'][ways] = {
            'case_count': len(results),
            'mean_pr_auc': np.mean(pr_aucs),
            'std_pr_auc': np.std(pr_aucs),
            'min_pr_auc': np.min(pr_aucs),
            'max_pr_auc': np.max(pr_aucs)
        }
    
    # 保存全局報告
    global_path = os.path.join(data_dir, 'global_pr_summary.json')
    with open(global_path, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    # 創建可讀的全局文本報告
    txt_path = os.path.join(data_dir, 'global_pr_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Global PR-AUC Summary Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Cases Across All Ways: {global_summary['total_cases']}\n")
        f.write(f"Valid Cases: {global_summary['valid_cases']}\n\n")
        
        f.write("Summary by Way Number:\n")
        for ways, stats in global_summary['by_ways'].items():
            f.write(f"  {ways}-way:\n")
            f.write(f"    Cases: {stats['case_count']}\n")
            f.write(f"    Mean PR-AUC: {stats['mean_pr_auc']:.4f}\n")
            f.write(f"    Std PR-AUC:  {stats['std_pr_auc']:.4f}\n")
            f.write(f"    Range: {stats['min_pr_auc']:.4f} - {stats['max_pr_auc']:.4f}\n\n")
    
    print(f"Saved global summary: {global_path}")
    print(f"Saved global text report: {txt_path}")

def plot_pr_curves_sklearn(true_labels, similarities, unique_classes, save_path, title="Precision-Recall Curves"):
    """為指定的way數繪製PR曲線，增加title參數"""
    plt.figure(figsize=(10, 8))
    
    for class_label in unique_classes:
        binary_true = [1 if label == class_label else 0 for label in true_labels]
        
        if sum(binary_true) > 0 and sum(binary_true) < len(binary_true):
            # 找到該類別對應的相似度分數列索引
            try:
                class_idx = unique_classes.index(class_label)
                class_similarities = similarities[:, class_idx]
                precision, recall, _ = precision_recall_curve(binary_true, class_similarities)
                ap = average_precision_score(binary_true, class_similarities)
                plt.plot(recall, precision, label=f'Class {class_label} (AP={ap:.3f})')
            except IndexError:
                print(f"Warning: Cannot find similarities for class {class_label}")
                continue
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR curves saved to: {save_path}")

def calculate_per_case_metrics(way_list, case_len, all_results):
    """為每個測試案例單獨計算PR-AUC"""
    case_metrics = {}
    
    for ways in way_list:
        case_metrics[f'{ways}_ways'] = []
        
        for case_idx in range(case_len):
            # 獲取該案例的結果
            case_true_labels = all_results[f'{ways}_ways']['cases'][case_idx]['true_labels']
            case_predictions = all_results[f'{ways}_ways']['cases'][case_idx]['predictions'] 
            case_similarities = all_results[f'{ways}_ways']['cases'][case_idx]['similarities']
            
            # 計算該案例的PR-AUC（應該只有5個類別）
            unique_classes = sorted(list(set(case_true_labels)))
            if len(unique_classes) == ways:  # 確認是正確的way數
                pr_aucs = calculate_pr_auc_multiclass(case_true_labels, case_similarities, unique_classes)
                avg_pr_auc = np.mean(pr_aucs)
                case_metrics[f'{ways}_ways'].append(avg_pr_auc)
        
        # 計算該way的平均PR-AUC
        if case_metrics[f'{ways}_ways']:
            print(f"{ways}-way Average PR-AUC: {np.mean(case_metrics[f'{ways}_ways']):.4f}")
            print(f"  Per-case PR-AUCs: {[f'{x:.3f}' for x in case_metrics[f'{ways}_ways']]}")
    
    return case_metrics

def calculate_pr_auc_multiclass(true_labels, similarities, unique_classes):
    
    """計算多類別的 PR-AUC，similarities 應為 [n_samples, n_classes]"""
    
    # 檢查輸入格式
    y_true = np.array(true_labels)
    y_score = np.array(similarities)
    
    print(f"Input shapes: y_true={y_true.shape}, y_score={y_score.shape}")
    print(f"Unique classes: {unique_classes}")
    
    # 確保 y_score 是 2D 且第二維等於類別數
    if len(y_score.shape) != 2:
        raise ValueError(f"similarities 必須是 2D 數組，當前 shape: {y_score.shape}")
    
    if y_score.shape[1] != len(unique_classes):
        raise ValueError(f"similarities 第二維({y_score.shape[1]})必須等於類別數({len(unique_classes)})")
    
    # 將標籤轉換為二進制格式
    y_true_binary = label_binarize(y_true, classes=unique_classes)
    
    pr_aucs = []
    for i, class_name in enumerate(unique_classes):
        binary_true = y_true_binary[:, i]
        class_scores = y_score[:, i]
        
        # 檢查是否有正負樣本
        if np.sum(binary_true) > 0 and np.sum(binary_true) < len(binary_true):
            ap = average_precision_score(binary_true, class_scores)
            pr_aucs.append(ap)
            print(f"Class {class_name}: AP = {ap:.4f}, positive samples = {np.sum(binary_true)}")
        else:
            pr_aucs.append(0.0)
            print(f"Class {class_name}: 跳過（只有一種標籤）")
    
    return pr_aucs

def check_pr_auc_inputs(true_labels, similarities, unique_classes):
    import numpy as np

    y_true = np.array(true_labels)
    y_score = np.array(similarities)

    print("true_labels shape:", y_true.shape)
    print("similarities shape:", y_score.shape)
    print("unique_classes:", unique_classes)

    # 檢查 similarities shape
    if len(y_score.shape) != 2 or y_score.shape[1] != len(unique_classes):
        print("錯誤：similarities 應為 [n_samples, n_classes]，目前 shape =", y_score.shape)
    else:
        print("similarities 格式正確")

    # 檢查 true_labels 是否都在 unique_classes
    if not set(y_true).issubset(set(unique_classes)):
        print("錯誤：true_labels 有不在 unique_classes 的值")
    else:
        print("true_labels 格式正確")

    # 檢查每個 class 樣本數
    for cls in unique_classes:
        count = np.sum(y_true == cls)
        print(f"Class {cls} 樣本數: {count}")

def save_confusion_matrix_plot(cm, title, save_path, class_labels=None):
    """Save confusion matrix as a heatmap plot."""
    plt.figure(figsize=(10, 8))
    
    # Create labels for the plot
    if class_labels is not None:
        labels = [str(label) for label in class_labels]
    else:
        labels = [str(i) for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_confusion_matrix_data(cm, title, save_dir, class_labels=None):
    """
    Save confusion matrix data in multiple formats (CSV, TXT) for later analysis.
    
    Args:
        cm: Confusion matrix (numpy array)
        title: Title for the confusion matrix (e.g., "5_ways", "overall")
        save_dir: Directory to save the files
        class_labels: List of class labels
    
    Returns:
        dict: Paths of saved files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Clean title for filename
    clean_title = title.replace(' ', '_').replace('-', '_').lower()
    
    # Save as CSV
    csv_path = os.path.join(save_dir, f"confusion_matrix_{clean_title}.csv")
    if class_labels is not None:
        labels = [str(label) for label in class_labels]
    else:
        labels = [f"class_{i}" for i in range(len(cm))]
    
    # Create DataFrame with proper labels
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(csv_path)
    
    # Save as detailed text file
    txt_path = os.path.join(save_dir, f"confusion_matrix_{clean_title}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Confusion Matrix - {title}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write matrix with labels
        f.write("Rows = True Labels, Columns = Predicted Labels\n\n")
        
        # Write header
        f.write("True\\Pred\t")
        for label in labels:
            f.write(f"{label}\t")
        f.write("\n")
        
        # Write matrix data
        for i, true_label in enumerate(labels):
            f.write(f"{true_label}\t\t")
            for j in range(len(labels)):
                f.write(f"{cm[i][j]}\t")
            f.write("\n")
        
        # Write summary statistics
        f.write(f"\n\nSummary Statistics:\n")
        f.write(f"Total predictions: {cm.sum()}\n")
        f.write(f"Correct predictions: {cm.diagonal().sum()}\n")
        f.write(f"Overall accuracy: {cm.diagonal().sum() / cm.sum():.4f}\n")
        
        # Per-class statistics
        f.write(f"\nPer-class Statistics:\n")
        for i, label in enumerate(labels):
            true_positives = cm[i][i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write(f"Class {label}:\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1-score: {f1:.4f}\n\n")
    
    # Save as numpy array for exact reconstruction
    npy_path = os.path.join(save_dir, f"confusion_matrix_{clean_title}.npy")
    np.save(npy_path, cm)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f"confusion_matrix_{clean_title}_metadata.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"Confusion Matrix Metadata - {title}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Shape: {cm.shape}\n")
        f.write(f"Class labels: {labels}\n")
        f.write(f"Total samples: {cm.sum()}\n")
        f.write(f"Number of classes: {len(labels)}\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
    
    print(f"Confusion matrix data saved:")
    print(f"  CSV: {csv_path}")
    print(f"  TXT: {txt_path}")
    print(f"  NPY: {npy_path}")
    print(f"  Metadata: {metadata_path}")
    
    return {
        'csv': csv_path,
        'txt': txt_path,
        'npy': npy_path,
        'metadata': metadata_path
    }


def load_and_plot_confusion_matrix(data_dir, matrix_name, output_path=None, title=None):
    """
    Load confusion matrix from saved files and regenerate the plot.
    
    Args:
        data_dir: Directory containing saved confusion matrix files
        matrix_name: Name of the matrix file (without extension)
        output_path: Path to save the new plot (optional)
        title: Custom title for the plot (optional)
    
    Returns:
        numpy.ndarray: The loaded confusion matrix
    """
    # Load the numpy array
    npy_path = os.path.join(data_dir, f"{matrix_name}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Confusion matrix file not found: {npy_path}")
    
    cm = np.load(npy_path)
    
    # Load metadata to get class labels
    metadata_path = os.path.join(data_dir, f"{matrix_name}_metadata.txt")
    class_labels = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract class labels from metadata
            for line in content.split('\n'):
                if line.startswith('Class labels:'):
                    labels_str = line.split(':', 1)[1].strip()
                    # Parse the list string (handle both ['a', 'b'] and [a, b] formats)
                    try:
                        class_labels = eval(labels_str)
                    except:
                        class_labels = None
                    break
    
    # Generate the plot
    if output_path is None:
        output_path = os.path.join(data_dir, f"{matrix_name}_regenerated.png")
    
    if title is None:
        title = matrix_name.replace('_', ' ').title()
    
    save_confusion_matrix_plot(cm, title, output_path, class_labels)
    
    print(f"Confusion matrix plot regenerated: {output_path}")
    return cm


def save_all_confusion_matrices_data(metrics_dict, save_dir):
    """
    Save all confusion matrices from metrics_dict in multiple formats.
    
    Args:
        metrics_dict: Dictionary containing all metrics from group_test_with_metrics
        save_dir: Directory to save all the confusion matrix data
    
    Returns:
        dict: Summary of all saved files
    """
    saved_files = {}
    
    for way_name, metrics in metrics_dict.items():
        if metrics.get('confusion_matrix') is not None:
            cm = metrics['confusion_matrix']
            class_labels = metrics.get('unique_classes', None)
            
            # Save the confusion matrix data
            file_paths = save_confusion_matrix_data(cm, way_name, save_dir, class_labels)
            saved_files[way_name] = file_paths
    
    # Create an index file listing all saved matrices
    index_path = os.path.join(save_dir, "confusion_matrices_index.txt")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("Confusion Matrices Index\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write(f"Total matrices saved: {len(saved_files)}\n\n")
        
        for way_name, files in saved_files.items():
            f.write(f"{way_name}:\n")
            for file_type, file_path in files.items():
                f.write(f"  {file_type.upper()}: {os.path.basename(file_path)}\n")
            f.write("\n")
        
        f.write("\nUsage Examples:\n")
        f.write("1. Load and plot from Python:\n")
        f.write("   from enhanced_testing import load_and_plot_confusion_matrix\n")
        f.write("   cm = load_and_plot_confusion_matrix('data_dir', 'matrix_name')\n\n")
        f.write("2. Read CSV in Excel/Python:\n")
        f.write("   import pandas as pd\n")
        f.write("   df = pd.read_csv('confusion_matrix_*.csv', index_col=0)\n")
    
    print(f"\nAll confusion matrices saved to: {save_dir}")
    print(f"Index file created: {index_path}")
    return saved_files


def save_metrics_to_csv(metrics_dict, save_path):
    """Save detailed metrics to CSV file."""
    rows = []
    
    for way_name, metrics in metrics_dict.items():
        if metrics['avg_pr_auc'] is not None:
            row = {
                'way': way_name,
                'avg_pr_auc': metrics['avg_pr_auc'],
                'num_classes': len(metrics['unique_classes']) if 'unique_classes' in metrics else 0
            }
            
            # Add per-class PR-AUC
            for i, class_auc in enumerate(metrics['pr_auc_per_class']):
                row[f'class_{i}_pr_auc'] = class_auc
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Detailed metrics saved to: {save_path}")


def Testing_Loop_with_predictions(model_num, dataset, net_name, model_path, testing_data_path, pr_dataset, pr_data_path="", encoder_path=None):
    """
    更新後的 Testing_Loop_with_predictions，現在會返回實際的預測結果
    """
    
    # 直接呼叫修改後的 Testing_Loop_0
    result = Testing_Loop_0(model_num, dataset, net_name, model_path, testing_data_path, pr_dataset, pr_data_path, encoder_path)
    
    # 檢查格式
    if "similarities" in result:
        similarities = result["similarities"]
        if len(similarities.shape) == 2:
            print(f"成功收集了 {len(result['predictions'])} 個預測結果")
            print(f"Similarities shape: {similarities.shape}")
            print(f"預測類別範例: {result['predictions'][:3]}")
            print(f"真實類別範例: {result['true_labels'][:3]}")
        else:
            print(f"警告：similarities 格式不正確，shape: {similarities.shape}")
    else:
        print("警告：沒有找到 similarities")
    
    return result

import glob
import re

def find_best_model(best_model_dir):
    """
    自動找到最佳模型檔案並解析參數
    
    Args:
        best_model_dir: 最佳模型資料夾路径 (e.g., "10_si_best_model")
    
    Returns:
        dict: 包含 model_path, epoch, batch_num 的字典
    """
    # 搜尋模型檔案 (.pth)
    model_pattern = os.path.join(best_model_dir, "*.pth")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"在 {best_model_dir} 中找不到模型檔案")
    
    if len(model_files) > 1:
        print(f"找到多個模型檔案: {model_files}")
        print("使用第一個檔案")
    
    model_path = model_files[0]
    filename = os.path.basename(model_path)
    
    # 解析檔案名稱 (假設格式為: 5ways_best_model_40ep_2690batch.pth)
    pattern = r"(\d+)ways_best_model_(\d+)ep_(\d+)batch\.pth"
    match = re.search(pattern, filename)
    
    if match:
        ways = int(match.group(1))
        epoch = int(match.group(2))
        batch_num = int(match.group(3))
    else:
        # 如果檔案名稱格式不匹配，嘗試其他可能的格式
        # 或者讓使用者手動指定
        print(f"無法解析檔案名稱格式: {filename}")
        print("請確認檔案名稱格式，或手動指定參數")
        return None
    
    return {
        'model_path': model_path,
        'epoch': epoch,
        'batch_num': batch_num,
        'ways': ways
    }

if __name__ == "__main__":
    print("Enhanced Testing Module Loaded")
    print("To use confusion matrix and PR-AUC calculation:")
    print("1. Import this module: from enhanced_testing import group_test_with_metrics")
    print("2. Replace your group_test calls with group_test_with_metrics")
    print("3. Modify Testing_Loop_0 to collect predictions, true_labels, and probabilities") 