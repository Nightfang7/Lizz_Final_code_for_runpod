import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torchvision.datasets as datasets

from clustering import DBSCAN_clustering_with_auto_emb_new
from networks import Autoencoder_Conv
from utils import (AutoEncoder_Training_Loop, generate_training_prototype,
                   generate_testing_prototype_new, re_group_train_data, Training_Loop)
from enhanced_testing import find_best_model, group_test_with_metrics
from dataset import SiameseNetworkDataset_1


def complete_training_loop_1(autoencoder_training_data_path, autoencoder_save_model_path,
                             autoencoder_save_img_base_path,
                             training_data_cluster_target, training_clustered_inter_dir,
                             training_clustered_best_dir,
                             train_prototype_dir, regrouped_target_dir,
                             test_cluster_target_dir, testing_clustered_inter_dir,
                             testing_clustered_best_dir,
                             save_best_si_model_dir, save_inter_si_model_dir,
                             si_epoch_num, vali_data_dir,
                             way_list, net_name, csv_save_dir, case_len, net_type, shots,
                             test_prototype_dest_dir,
                             proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                             result_csv_dir, transformation,
                             model_num=2, auto_epoch_num=100, random_seed=42, encoder_path=None,
                             early_stopping_enable=False, early_stopping_patience=5,
                             early_stopping_min_delta=0.0, early_stopping_monitor='accuracy',
                             save_pr_curve_images=True, validate_every_epochs=20):
    '''
    Clustering: DBSCAN with autoencoder embeddings
    Network: SiameseNetwork_autoencoder_Based (pretrained autoencoder)
    Steps:
      1. Train Autoencoder
      2. Cluster training data (DBSCAN)
      3. Generate training prototypes
      4. Re-cluster prototypes, re-group training data
      5. Cluster testing data (DBSCAN), generate testing prototypes
      6. Train Siamese network
      7. Test Siamese network
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.path.join(os.path.normpath(test_cluster_target_dir), i)
                               for i in cluster_target_folder_names]

    # 1. Train Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path,
                              transformation=transformation)

    # 2. Cluster training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path, weights_only=True))
    net.eval()

    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15, min_samples=1, drop_noise=False)
    train_best_clustered_path = best_cluster_path

    # 3. Generate training prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path,
                                dest_dataset_path=train_prototype_dir)

    # 4. Re-cluster prototypes and re-group training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13, min_samples=1, drop_noise=False)

    re_group_train_data(source_dir=train_best_clustered_path,
                        info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)

    # 5. Cluster testing data and generate testing prototypes
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.path.join(os.path.normpath(testing_clustered_inter_dir),
                                                   cluster_target_folder_names[i])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.path.join(os.path.normpath(testing_clustered_best_dir),
                                                   cluster_target_folder_names[i])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.path.join(os.path.normpath(test_prototype_dest_dir),
                                                       cluster_target_folder_names[i])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
            dataset_path=cluster_target_folders[i],
            transformation=transformation,
            inter_dest_dir=single_case_inter_dest_dir,
            best_dest_dir=single_case_inter_best_dir,
            net=net, save_inter=False,
            desig_eps=15, min_samples=1)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                       dest_dataset_path=single_case_prototype_dest_dir)

    # 6. Train Siamese network
    print("====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(
        model_num=model_num, net_name=net_name,
        dataset=siamese_train_dataset,
        save_best_model_dir=save_best_si_model_dir,
        save_inter_model_dir=save_inter_si_model_dir,
        epoch_num=si_epoch_num, way_list=[5],
        vali_data_dir=vali_data_dir,
        proto_data_dir=test_prototype_dest_dir,
        csv_save_dir=csv_save_dir,
        case_len=case_len, net_type=net_type, encoder_path=encoder_path,
        early_stopping_enable=True,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=0.0,
        early_stopping_monitor='accuracy',
        validate_every_epochs=validate_every_epochs)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    # 7. Test Siamese network
    print("====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")
    best_model_dir = save_best_si_model_dir
    model_info = find_best_model(best_model_dir)
    if model_info is None:
        return None

    print(f"找到模型: {model_info['model_path']}")
    print(f"Epoch: {model_info['epoch']}, Batch: {model_info['batch_num']}, Ways: {model_info['ways']}")

    result_df, avg_df, metrics_dict = group_test_with_metrics(
        model_num=model_num,
        way_list=way_list,
        case_len=case_len,
        epoch=model_info['epoch'],
        model_path=model_info['model_path'],
        batch_num=model_info['batch_num'],
        net_name=net_name,
        net_type=net_type,
        test_data_dir=vali_data_dir,
        proto_data_dir=test_prototype_dest_dir,
        save_metrics_dir=result_csv_dir,
        random_seed=random_seed,
        encoder_path=encoder_path,
        save_pr_curve_images=save_pr_curve_images,
        shots=shots)

    print("Accuracy Results:")
    print(result_df)
    print("\nAverage Accuracy:")
    print(avg_df)
    print("\nConfusion Matrix and PR-AUC Metrics:")
    for way, metrics in metrics_dict.items():
        print(f"\n{way}:")
        if metrics['avg_pr_auc'] is not None:
            print(f"  Average PR-AUC: {metrics['avg_pr_auc']:.4f}")
            print(f"  Confusion Matrix Shape: {metrics['confusion_matrix'].shape}")

    result_df.to_csv(os.path.join(os.path.normpath(result_csv_dir), "results.csv"))
    avg_df.to_csv(os.path.join(os.path.normpath(result_csv_dir), "avg.csv"))
