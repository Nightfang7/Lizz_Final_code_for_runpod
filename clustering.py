import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import shutil
import torch
from collections import Counter
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from dataset import DBSCANDataset


def DBSCAN_clustering_with_auto_emb_new(dataset_path, transformation, inter_dest_dir, best_dest_dir, net=None,
                                        save_inter=True, desig_eps=None, min_samples=1, drop_noise=False):
    images_groups = os.listdir(dataset_path)
    prototypes = {class_idx: [] for class_idx in images_groups}

    for i in range(len(images_groups)):
        images_group_path = os.path.join(dataset_path, images_groups[i])
        folder_dataset = datasets.ImageFolder(root=images_group_path)
        DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset, transform=transformation)
        DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=16, shuffle=False, num_workers=0)
        images_group_len = len(DBSCAN_dataset)
        print(images_group_len)

        embeddings_list = []
        for img_0, label_0, path_0 in DBSCAN_dataloader:
            with torch.no_grad():
                outputs = net.encoder(img_0.cuda())
            for emb in outputs:
                embeddings_list.append(emb.detach().cpu().numpy().flatten())

        embeddings_array = np.array(embeddings_list)

        EPS = desig_eps
        if EPS is None:
            cluster_results = []
            for m in range(1, 100):
                EPS = float(m)
                clustering = DBSCAN(eps=EPS, min_samples=2)
                SC_result = clustering.fit_predict(embeddings_array)
                print("Clustering Results!!!!!!!!!! ", SC_result)
                cluster = SC_result
                Labels = list(set(SC_result))
                Labels_len = len(list(set(SC_result)))
                noise_len = Counter(SC_result)[-1]
                cluster_results.append((EPS, Labels, Labels_len, noise_len))
                print("eps: ", EPS, "Labels: ", Labels, "Labels_len: ", Labels_len,
                      "SC_result_len: ", len(SC_result), "noise_len: ", noise_len)
                if save_inter:
                    clustered_path = os.path.join(inter_dest_dir, f"{EPS}eps")
                    if not os.path.exists(clustered_path):
                        os.mkdir(clustered_path)
                    try:
                        Labels_copy = Labels.copy()
                        Labels_copy.remove(-1)
                        Labels_drop_len = len(Labels_copy)
                    except Exception:
                        Labels_drop_len = Labels_len
                    for q in range(Labels_drop_len):
                        subclass_path = os.path.join(clustered_path, f'{images_groups[i]}_{q}')
                        if not os.path.exists(subclass_path):
                            os.mkdir(subclass_path)
                    noise_count = 1
                    for q in range(len(SC_result)):
                        image, label, original_path = DBSCAN_dataset[q]
                        path = os.path.basename(original_path)
                        if SC_result[q] != -1:
                            target_path = os.path.join(clustered_path,
                                                       f'{images_groups[i]}_{Labels.index(SC_result[q])}', path)
                            shutil.copyfile(original_path, target_path)
                        else:
                            print("noise count: ", noise_count)
                            subclass_path = os.path.join(clustered_path,
                                                         f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise')
                            if not os.path.exists(subclass_path):
                                os.mkdir(subclass_path)
                            target_path = os.path.join(clustered_path,
                                                       f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise',
                                                       path)
                            shutil.copyfile(original_path, target_path)
                            noise_count += 1

            max_index = [i[2] - (i[3] / 3) for i in cluster_results].index(
                max([i[2] - (i[3] / 3) for i in cluster_results]))
            EPS = cluster_results[max_index][0]

        clustered_path = os.path.join(best_dest_dir, "final_clustered_dataset")
        best_cluster_path = clustered_path
        if not os.path.exists(clustered_path):
            os.mkdir(clustered_path)

        clustering = DBSCAN(eps=EPS, min_samples=min_samples)
        SC_result = clustering.fit_predict(embeddings_array)

        unique_clusters = np.unique(SC_result)
        cluster_prototypes = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            cluster_points = embeddings_array[SC_result == cluster_id]
            cluster_centroid = np.mean(cluster_points, axis=0)
            cluster_prototypes.append(cluster_centroid)
        prototypes[images_groups[i]] = cluster_prototypes
        print(f"PROTOTYPES: {images_groups[i]} -> {len(cluster_prototypes)} clusters")

        cluster = SC_result
        Labels = list(set(SC_result))
        Labels_len = len(list(set(SC_result)))
        try:
            Labels_copy = Labels.copy()
            Labels_copy.remove(-1)
            Labels_drop_len = len(Labels_copy)
        except Exception:
            Labels_drop_len = Labels_len

        for j in range(Labels_drop_len):
            subclass_path = os.path.join(clustered_path, f'{images_groups[i]}_{j}_{EPS}eps')
            if not os.path.exists(subclass_path):
                os.mkdir(subclass_path)

        noise_count = 1
        for j in range(len(SC_result)):
            image, label, original_path = DBSCAN_dataset[j]
            path = os.path.basename(original_path)
            if SC_result[j] != -1:
                target_path = os.path.join(clustered_path,
                                           f'{images_groups[i]}_{Labels.index(SC_result[j])}_{EPS}eps', path)
                shutil.copyfile(original_path, target_path)
            else:
                if drop_noise:
                    continue
                subclass_path = os.path.join(clustered_path,
                                             f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise')
                if not os.path.exists(subclass_path):
                    os.mkdir(subclass_path)
                target_path = os.path.join(clustered_path,
                                           f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise',
                                           path)
                shutil.copyfile(original_path, target_path)
                noise_count += 1

    return best_cluster_path
