import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import re
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN

from dataset import AutoEncoderDataset, SiameseNetworkDataset_1, PrototypeDataset
from networks import (Autoencoder_Conv, SiameseNetwork_autoencoder_Based,
                      SiameseNetwork_only_autoencoder)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def AutoEncoder_Training_Loop(epoch_num, data_path, save_model_path, save_img_base_path, transformation):
    import matplotlib.pyplot as plt
    folder_dataset = datasets.ImageFolder(root=data_path)
    autoencoder_dataset = AutoEncoderDataset(imageFolderDataset=folder_dataset, transform=transformation)
    autoencoder_dataloader = DataLoader(autoencoder_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = Autoencoder_Conv()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    outputs = []
    for epoch in range(epoch_num):
        for (img, _, _) in autoencoder_dataloader:
            recon = model(img)
            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))

    torch.save(model.state_dict(), save_model_path)

    for k in range(0, epoch_num, 5):
        plt.figure(figsize=(9, 2))
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            plt.imshow(np.transpose(item, (1, 2, 0)))
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(np.transpose(item, (1, 2, 0)))
        save_img_path = os.path.join(os.path.normpath(save_img_base_path), f"Epoch_{k}.png")
        plt.savefig(save_img_path)
        plt.close()


def generate_training_prototype(target_dataset_path, dest_dataset_path):
    TARGETFOLDER = target_dataset_path
    DESTFOLDER = dest_dataset_path
    if not os.path.exists(DESTFOLDER):
        os.mkdir(DESTFOLDER)
    class_folder_name = os.listdir(TARGETFOLDER)
    print(class_folder_name)
    for i in range(len(class_folder_name)):
        image_folder = os.path.join(TARGETFOLDER, class_folder_name[i])
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        sum_image = None
        for image_file in image_files:
            img = cv2.imread(os.path.join(image_folder, image_file))
            if sum_image is None:
                sum_image = np.zeros_like(img, dtype=np.float64)
            sum_image += img.astype(np.float64)
        average_image = (sum_image / len(image_files)).astype(np.uint8)
        class_path = os.path.join(DESTFOLDER, 'classes', 'class')
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        cv2.imwrite(os.path.join(class_path, f'{class_folder_name[i]}_average_image.png'), average_image)


def generate_testing_prototype_new(target_dataset_path, dest_dataset_path):
    TARGETFOLDER = target_dataset_path
    DESTFOLDER = dest_dataset_path
    if not os.path.exists(DESTFOLDER):
        os.mkdir(DESTFOLDER)
    class_folder_name = os.listdir(TARGETFOLDER)
    print(class_folder_name)
    for i in range(len(class_folder_name)):
        image_folder = os.path.join(TARGETFOLDER, class_folder_name[i])
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        sum_image = None
        for image_file in image_files:
            img = cv2.imread(os.path.join(image_folder, image_file))
            if sum_image is None:
                sum_image = np.zeros_like(img, dtype=np.float64)
            sum_image += img.astype(np.float64)
        average_image = (sum_image / len(image_files)).astype(np.uint8)
        class_path = os.path.join(DESTFOLDER, class_folder_name[i])
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        cv2.imwrite(os.path.join(class_path, f'{class_folder_name[i]}_average_image.png'), average_image)


def re_group_train_data(source_dir, info_dir, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    class_list = os.listdir(source_dir)
    print("class_list: ", class_list)
    group_list = os.listdir(info_dir)
    print("group_list: ", group_list)
    for i in range(len(group_list)):
        group_list_path = os.path.join(info_dir, group_list[i])
        class_img_in_group = os.listdir(group_list_path)
        classes_in_group = ["_".join(f.split("_")[0:-2]) for f in class_img_in_group]
        # 顯示名稱只取 family 部分（去掉 _N_Xeps 後綴），避免 Windows 路徑過長
        display_names = ["_".join(c.split("_")[:-2]) if len(c.split("_")) > 2 else c for c in classes_in_group]
        composed_name = "-".join(display_names)
        new_class_name = f"group_{i:04d}"
        print(f"new_class_name: {new_class_name} (composed of: {composed_name})")
        # DEBUG: 驗證 source_dir 是否真的存在
        for dbg_cls in classes_in_group:
            dbg_path = os.path.join(source_dir, dbg_cls)
            print(f"  [DEBUG] classes_in_group entry: '{dbg_cls}' -> exists={os.path.exists(dbg_path)}")
        new_class_path = os.path.join(target_dir, new_class_name)
        if int(group_list[i].split("_")[1]) != len(group_list) - 1:
            already_exists = os.path.exists(new_class_path)
            print(f"  [DEBUG] new_class_path='{new_class_path}' already_exists={already_exists}")
            if not already_exists:
                os.mkdir(new_class_path)
            for j in range(len(classes_in_group)):
                source_class_dir = os.path.join(source_dir, classes_in_group[j])
                source_class_img = os.listdir(source_class_dir)
                for img_name in source_class_img:
                    shutil.copyfile(os.path.join(source_class_dir, img_name),
                                    os.path.join(new_class_path, img_name))
        else:
            for j in range(len(classes_in_group)):
                source_class_dir = os.path.join(source_dir, classes_in_group[j])
                source_class_img = os.listdir(source_class_dir)
                target_class_path = os.path.join(target_dir, classes_in_group[j])
                if not os.path.exists(target_class_path):
                    os.mkdir(target_class_path)
                for img_name in source_class_img:
                    shutil.copyfile(os.path.join(source_class_dir, img_name),
                                    os.path.join(target_class_path, img_name))


def get_class_name_from_path(file_path):
    return os.path.basename(os.path.dirname(file_path))


def get_class_prefix_from_path(file_path):
    class_name = get_class_name_from_path(file_path)
    return class_name.split('_')[0]


def Testing_Loop_0(model_num, dataset, net_name, model_path, testing_data_path, pr_dataset, pr_data_path="", encoder_path=None):
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    auto_net = SiameseNetwork_only_autoencoder(encoder_path=encoder_path).cuda()
    auto_net.eval()

    net = SiameseNetwork_autoencoder_Based(encoder_path=encoder_path).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    predictions = []
    true_labels = []
    all_similarities = []

    with torch.inference_mode():
        if model_num == 1:
            print("INTO MODEL NUM 1")
            correct_count = 0
            total = 0
            class_list = os.listdir(testing_data_path)
            print(class_list)

            for j, (x0, _, _, label1, _, L1, _) in enumerate(test_dataloader, 0):
                prob_list = []
                a_prob_list = []
                a_proto_vect = []
                class_distances = []

                for i in range(len(class_list)):
                    pr_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                    dataiter = iter(pr_dataloader)
                    while True:
                        try:
                            x1, _, _, label2, _, L2, _ = next(dataiter)
                        except StopIteration:
                            pr_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                            dataiter = iter(pr_dataloader)
                            continue
                        if get_class_name_from_path(L2[0]) == class_list[i]:
                            output1, output2 = net(x0.cuda(), x1.cuda())
                            euclidean_distance = F.pairwise_distance(output1, output2)
                            class_distances.append(euclidean_distance.item())
                            prob_list.append((euclidean_distance.item(),
                                              (get_class_prefix_from_path(L1[0]),
                                               get_class_prefix_from_path(L2[0]))))
                            a_output1, a_output2 = auto_net(x0.cuda(), x1.cuda())
                            a_euclidean_distance = F.pairwise_distance(a_output1, a_output2)
                            a_prob_list.append((a_euclidean_distance.item(),
                                                (get_class_prefix_from_path(L1[0]),
                                                 get_class_prefix_from_path(L2[0]))))
                            a_proto_vect.append(a_output2.detach().cpu().numpy().flatten())
                            break

                a_proto_array = np.array(a_proto_vect)
                clustering = DBSCAN(eps=8, min_samples=2)
                SC_result = clustering.fit_predict(a_proto_array)
                sc_list = list(SC_result)
                try:
                    sc_list.remove(-1)
                except ValueError:
                    pass
                use_auto = len(list(set(sc_list))) > 0

                true_class = get_class_prefix_from_path(L1[0])

                if use_auto:
                    score = [a_prob_list[i][0] for i in range(len(a_prob_list))]
                    min_score_index = score.index(min(score))
                    predicted_class = a_prob_list[min_score_index][1][1]
                    if a_prob_list[min_score_index][1][0] == a_prob_list[min_score_index][1][1]:
                        correct_count += 1
                else:
                    score = [prob_list[i][0] for i in range(len(prob_list))]
                    min_score_index = score.index(min(score))
                    predicted_class = prob_list[min_score_index][1][1]
                    if prob_list[min_score_index][1][0] == prob_list[min_score_index][1][1]:
                        correct_count += 1

                total = j + 1
                predictions.append(predicted_class)
                true_labels.append(true_class)
                all_similarities.append([-d for d in class_distances])

            print("correct_count: ", correct_count)
            print("total: ", total)
            print("accuracy: ", correct_count / total)

        else:
            correct_count = 0
            total = 0
            prototype_folders = sorted(os.listdir(pr_data_path))

            main_classes = {}
            for folder in prototype_folders:
                main_class = re.sub(r'_\d+_\d+eps$', '', folder)
                if main_class not in main_classes:
                    main_classes[main_class] = []
                main_classes[main_class].append(folder)

            main_class_names = sorted(main_classes.keys())
            print(f"主類別: {main_class_names}")
            print(f"每個類別的原型: {main_classes}")

            for j, (x0, _, _, label1, _, L1, _) in enumerate(test_dataloader, 0):
                main_class_distances = []

                for main_class in main_class_names:
                    class_prototype_distances = []
                    for prototype_folder in main_classes[main_class]:
                        pr_dataloader = DataLoader(pr_dataset, batch_size=1, shuffle=True)
                        dataiter = iter(pr_dataloader)
                        while True:
                            try:
                                x1, label2, L2 = next(dataiter)
                            except StopIteration:
                                pr_dataloader = DataLoader(pr_dataset, batch_size=1, shuffle=True)
                                dataiter = iter(pr_dataloader)
                                continue
                            if get_class_name_from_path(L2[0]) == prototype_folder:
                                output1, output2 = net(x0.cuda(), x1.cuda())
                                a_output1, a_output2 = auto_net(x0.cuda(), x1.cuda())
                                s_distance_1 = F.pairwise_distance(F.normalize(output1.unsqueeze(0), p=1, dim=1),
                                                                    F.normalize(output2.unsqueeze(0), p=1, dim=1))
                                s_distance_2 = F.pairwise_distance(F.normalize(a_output1.unsqueeze(0), p=1, dim=1),
                                                                    F.normalize(a_output2.unsqueeze(0), p=1, dim=1))
                                s_euclidean_distance = torch.stack([s_distance_1 * 2.5, s_distance_2])
                                final_distance = s_euclidean_distance.sum().item()
                                class_prototype_distances.append(final_distance)
                                break

                    if class_prototype_distances:
                        main_class_distances.append(min(class_prototype_distances))
                    else:
                        main_class_distances.append(float('inf'))

                min_distance_idx = np.argmin(main_class_distances)
                predicted_class = main_class_names[min_distance_idx]
                true_class_raw = get_class_prefix_from_path(L1[0])
                true_class = re.sub(r'_\d+_\d+eps$', '', true_class_raw)

                if predicted_class == true_class:
                    correct_count += 1
                total = j + 1
                predictions.append(predicted_class)
                true_labels.append(true_class)
                all_similarities.append([-d for d in main_class_distances])

                if j < 3:
                    print(f"Sample {j}: true={true_class}, pred={predicted_class}")
                    print(f"  主類別距離: {main_class_distances}")

            print("correct_count: ", correct_count)
            print("total: ", total)
            print("accuracy: ", correct_count / total)

    similarities_array = np.array(all_similarities)
    print(f"最終 similarities shape: {similarities_array.shape}")
    print(f"主類別: {main_class_names if model_num != 1 else sorted(os.listdir(testing_data_path))}")

    return {
        "correct_count": correct_count,
        "total": total,
        "accuracy": correct_count / total,
        "predictions": predictions,
        "true_labels": true_labels,
        "similarities": similarities_array,
        "class_names": main_class_names if model_num != 1 else sorted(os.listdir(testing_data_path))
    }


def group_test(model_num=1, way_list=[5, 6], case_len=10, epoch=0, model_path="", batch_num=0,
               net_name="", net_type="",
               test_data_dir="", test_data_postfix="",
               proto_data_dir="", proto_data_postfix="", encoder_path=None):
    result_df = pd.DataFrame()
    avg_df = pd.DataFrame()
    transformation = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    for j in range(len(way_list)):
        WAYS = way_list[j]
        cases_result = []
        for k in range(case_len):
            CASE = k + 1
            print(f'{WAYS} WAYS CASE {CASE}')
            TESTINGDATAPATH = os.path.join(test_data_dir, f"test_{WAYS}_{CASE}{test_data_postfix}")
            PROTODATAPATH = os.path.join(proto_data_dir, f"test_{WAYS}_{CASE}{proto_data_postfix}")
            print("TESTINGDATAPATH: ", TESTINGDATAPATH)
            print("PROTODATAPATH: ", PROTODATAPATH)

            folder_dataset_test = datasets.ImageFolder(root=TESTINGDATAPATH)
            siamese_test_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset_test,
                                                           transform=transformation)
            if model_num != 1:
                pr_folder_dataset_test = datasets.ImageFolder(root=PROTODATAPATH)
                pr_dataset = PrototypeDataset(imageFolderDataset=pr_folder_dataset_test,
                                              transform=transformation)
                pr_data_path = PROTODATAPATH
            else:
                pr_dataset = ""
                pr_data_path = ""

            result_dict = Testing_Loop_0(model_num=model_num, dataset=siamese_test_dataset,
                                         net_name=net_name, model_path=model_path,
                                         testing_data_path=TESTINGDATAPATH,
                                         pr_dataset=pr_dataset, pr_data_path=pr_data_path,
                                         encoder_path=encoder_path)
            cases_result.append(result_dict["accuracy"])

        col_name = f'{epoch}EPOCH{batch_num}BATCH-{WAYS}WAYS'
        result_df[col_name] = cases_result
        avg_df[col_name] = [np.average(cases_result)]
    print(result_df)
    print(avg_df)
    return result_df, avg_df


def Training_Loop(model_num, net_name, dataset, save_best_model_dir, save_inter_model_dir, epoch_num, way_list,
                  vali_data_dir, proto_data_dir, csv_save_dir, case_len, net_type, encoder_path=None,
                  early_stopping_enable=False, early_stopping_patience=3,
                  early_stopping_min_delta=0.0, early_stopping_monitor='accuracy',
                  validate_every_epochs=5):
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=64, drop_last=True, num_workers=0)
    net = SiameseNetwork_autoencoder_Based(encoder_path=encoder_path).cuda()

    criterion = ContrastiveLoss()
    base_lr = 0.0001
    optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=1e-4)
    try:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-6)
    except Exception:
        scheduler = None
    warmup_epochs = 5

    counter = []
    loss_history = []
    iteration_number = 0
    batch = 0
    result_df_list = []
    avg_df_list = []
    best_monitor_value = None
    best_epoch_seen = None
    patience_counter = 0
    stopped_early_at = None

    for epoch in range(epoch_num):
        if warmup_epochs is not None and epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
            for g in optimizer.param_groups:
                g['lr'] = base_lr * warmup_factor

        for i, (img0, img1, label, label_1, label_2, _, _) in enumerate(train_dataloader, 0):
            batch = i
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

        if scheduler is not None and (warmup_epochs is None or epoch >= warmup_epochs):
            try:
                scheduler.step()
            except Exception:
                pass

        if (validate_every_epochs is not None and validate_every_epochs > 0
                and epoch >= validate_every_epochs and (epoch % validate_every_epochs) == 0):
            model_postfix = f"_{epoch}ep_{batch}batch"
            save_inter_model_path = os.path.join(save_inter_model_dir, f"model{model_postfix}.pth")
            torch.save(net.state_dict(), save_inter_model_path)
            result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len,
                                           epoch=epoch, model_path=save_inter_model_path, batch_num=batch,
                                           net_name=net_name, net_type=net_type,
                                           test_data_dir=vali_data_dir, test_data_postfix="",
                                           proto_data_dir=proto_data_dir, proto_data_postfix="",
                                           encoder_path=encoder_path)
            result_df_list.append(result_df)
            avg_df_list.append(avg_df)

            if early_stopping_enable:
                try:
                    monitor_value = float(pd.to_numeric(
                        avg_df.select_dtypes(include=['number']).values.flatten(),
                        errors='coerce').mean())
                except Exception:
                    monitor_value = None

                improved = False
                if monitor_value is not None:
                    if best_monitor_value is None or (monitor_value - best_monitor_value) > early_stopping_min_delta:
                        best_monitor_value = monitor_value
                        best_epoch_seen = epoch
                        patience_counter = 0
                        improved = True
                    else:
                        patience_counter += 1

                print(f"[EarlyStopping] epoch={epoch}, monitor={monitor_value}, "
                      f"best={best_monitor_value}, patience={patience_counter}/{early_stopping_patience}")

                if not improved and patience_counter >= early_stopping_patience:
                    stopped_early_at = epoch
                    print(f"[EarlyStopping] Stop training early at epoch {epoch} (best epoch {best_epoch_seen})")
                    break

    result_final_df = pd.concat(result_df_list, axis=1)
    result_final_df.to_csv(os.path.join(csv_save_dir, "final_result.csv"))
    avg_final_df = pd.concat(avg_df_list, axis=1)
    avg_final_df.to_csv(os.path.join(csv_save_dir, "final_avg.csv"))

    if early_stopping_enable:
        try:
            es_info = {
                'enabled': True,
                'stopped_early_at': int(stopped_early_at) if stopped_early_at is not None else None,
                'best_epoch': int(best_epoch_seen) if best_epoch_seen is not None else None,
                'best_monitor_value': float(best_monitor_value) if best_monitor_value is not None else None,
                'monitor': early_stopping_monitor,
                'patience': int(early_stopping_patience),
                'min_delta': float(early_stopping_min_delta)
            }
            with open(os.path.join(csv_save_dir, 'early_stopping_info.json'), 'w', encoding='utf-8') as f:
                json.dump(es_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: failed to save early stopping info: {e}")

    best_epoch_list = []
    best_batch_list = []
    target_path_list = []
    for i in range(len(way_list)):
        col_list = [j for j in range(len(avg_final_df.columns)) if j % len(way_list) == i]
        avg_final_df_way = avg_final_df.iloc[:, col_list]
        best_result = avg_final_df_way.idxmax(axis=1)
        print("best_result: ", best_result[0])
        best_info = re.match(r"(\d+)EPOCH(\d+)BATCH-(\d+)WAYS", str(best_result[0]))
        best_epoch = best_info.group(1)
        best_batch = best_info.group(2)
        model_file_name = f"model_{best_epoch}ep_{best_batch}batch.pth"
        source_path = os.path.join(save_inter_model_dir, model_file_name)
        target_model_file_name = f"{way_list[i]}ways_best_model_{best_epoch}ep_{best_batch}batch.pth"
        if i == 0:
            target_path = os.path.join(save_best_model_dir, target_model_file_name)
            shutil.copyfile(source_path, target_path)
        best_epoch_list.append(best_epoch)
        best_batch_list.append(best_batch)
        target_path_list.append(target_path)

    return best_epoch_list, best_batch_list, target_path_list
