import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import random
from cruw import CRUW
from cruw.visualization.examples import show_dataset_rod2021
from cruw.mapping import ra2idx, idx2ra
import math
from cruw.eval import evaluate_rod2021
from rodnet.core.radar_processing import chirp_amp


def get_paths(seq_name, frame_id):
    image_path = os.path.join(data_root, 'sequences', 'train', seq_name,
                              dataset.sensor_cfg.camera_cfg['image_folder'],
                              '%010d.jpg' % frame_id)
    chirp_path = os.path.join(data_root, 'sequences', 'train', seq_name,
                              dataset.sensor_cfg.radar_cfg['chirp_folder'],
                              '%06d_0000.npy' % frame_id)
    anno_path = os.path.join(data_root, 'annotations', 'train', seq_name + '.txt')
    return image_path, chirp_path, anno_path

if __name__ == '__main__':
    data_root = r"E:\RADAR\mnt\disk1\CRUW\ROD2021"
    dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')

    # #Camera and radar configurations
    # print(dataset.sensor_cfg.camera_cfg)
    #
    # #Calibration parameters
    # print('camera intrinsics:')
    # print(dataset.sensor_cfg.calib_cfg['cam_calib']['2019_04_09']['cam_0']['camera_matrix'])
    # print('camera to radar translation:')
    # print(dataset.sensor_cfg.calib_cfg['t_cl2rh'])
    # #Object classes of interest
    # print(dataset.object_cfg.classes)
    #
    # #距离/方位角与 RF 图像中的索引之间的映射
    # #从绝对距离 (m) 和方位角 (rad) 映射到 RF 指数
    # rng = 5.0
    # azm = math.radians(30)  # degree to radians
    # rid, aid = ra2idx(rng, azm, dataset.range_grid, dataset.angle_grid)
    # print(rid, aid)
    # #从射频指数映射到绝对距离 (m) 和方位角 (rad) 注意：由于从绝对距离/方位角值到 RF 像素的离散化，距离和方位角不能绝对恢复。
    # rid = 20
    # aid = 95
    # rng, azm = idx2ra(rid, aid, dataset.range_grid, dataset.angle_grid)
    # print(rng, math.degrees(azm))
    #
    #数据可视化
    seq_name ='2019_05_09_PCMS002'
    #有问题  '2019_09_29_ONRD001''2019_09_29_ONRD002'
    #没问题 '2019_09_29_ONRD013' '2019_09_29_ONRD011' '2019_09_29_ONRD006''2019_09_29_ONRD005'
    #'2019_05_09_CM1S004' '2019_04_09_CMS1002' '2019_04_30_PCMS001'
    #骑车: 2019_04_09_BMS1000 2019_04_09_BMS1001 2019_04_09_BMS1002 2019_05_29_BM1S016 2019_05_29_BM1S017
    #car: 2019_04_09_CMS1002
    #骑车 car： 2019_05_09_BM1S008
    for i in range(0,900, 10):
        frame_id = i
        image_path, chirp_path, anno_path = get_paths(seq_name, frame_id)
        print(frame_id)
        show_dataset_rod2021(image_path, chirp_path, anno_path, dataset)
        plt.show()

    # conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1,padding=0,stride=2)
    # linear = nn.Linear(1, 2)
    # fig = plt.figure(figsize=(8, 8))
    #
    # # fig.add_subplot(1, 2, 1)
    # input_radar = np.load(r"E:\RADAR\mnt\disk1\CRUW\ROD2021\sequences\train\2019-09-16-12-55-51\000222.npy")
    #
    # # patch_size = 16
    # # mask_ratio = 0.75
    # #
    # # all_num = int(128 / patch_size)
    # # matrix = [i + 1 for i in range(all_num * all_num)]
    # # chosen = random.sample(matrix, int((mask_ratio) * all_num * all_num))
    # # # Mask
    # # for i in range(0, 128, 16):
    # #     for j in range(0, 128, 16):
    # #         if int((i + 1) / 2 + (j + 1) / patch_size + 1) in chosen:
    # #             input_radar[i:i + patch_size, j:j + patch_size, :] = 0
    #
    # # max_value = np.max(input_radar)
    # # min_value = np.min(input_radar)
    # # out = (input_radar - min_value) / (max_value - min_value)
    # chirp_amp_curr = chirp_amp(input_radar, 'ROD2021')
    # # chirp_amp_curr[chirp_amp_curr < 0] = 0
    # # chirp_amp_curr[chirp_amp_curr > 1] = 1
    # plt.imshow(chirp_amp_curr, origin='lower', aspect='auto')
    # plt.show()
    #
    # # #评估脚本
    # # submit_dir = '<SUBMISSION_DIR>'
    # # truth_dir = '<ANNOTATION_DIR>'
    # # evaluate_rod2021(submit_dir, truth_dir, dataset)