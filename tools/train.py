import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cruw import CRUW

from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.visualization import visualize_train_img
from Nets.T_RODNet import T_RODNet
from tools.loss_history import LossHistory

def parse_args():
    parser = argparse.ArgumentParser(description='Train T-RODNet.')

    parser.add_argument('--config', default=r"/home/long/PycharmProjects/T-RODNet-main/configs/config_T_Rodnet_win16.py",type=str, help='configuration file path')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--data_dir', type=str, default=r'/home/long/PycharmProjects/T-RODNet-main/tools/prepare_dataset/data', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='./checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the trained model')
    parser.add_argument('--save_memory', action="store_true", help="use customized dataloader to save memory")
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args

    # dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'])
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']
    if model_cfg['type'] == 'CDC':
        from rodnet.models import RODNetCDC as RODNet
    elif model_cfg['type'] == 'HG':
        from rodnet.models import RODNetHG as RODNet
    elif model_cfg['type'] == 'HGwI':
        from rodnet.models import RODNetHGwI as RODNet
    elif model_cfg['type'] == 'CDCv2':
        from rodnet.models import RODNetCDCDCN as RODNet
    elif model_cfg['type'] == 'HGv2':
        from rodnet.models import RODNetHGDCN as RODNet
    elif model_cfg['type'] == 'HGwIv2':
        from rodnet.models import RODNetHGwIDCN as RODNet
    else:
        raise NotImplementedError

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_model_path = args.log_dir

    # create / load models
    cp_path = None
    epoch_start = 0
    iter_start = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        cp_path = args.resume_from
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)
    else:
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)

    train_viz_path = os.path.join(model_dir, 'train_viz')
    if not os.path.exists(train_viz_path):
        os.makedirs(train_viz_path)

    writer = SummaryWriter(model_dir)
    save_config_dict = {
        'args': vars(args),
        'config_dict': config_dict,
    }
    config_json_name = os.path.join(model_dir, 'config-' + time.strftime("%Y%m%d-%H%M%S") + '.json')
    with open(config_json_name, 'w') as fp:
        json.dump(save_config_dict, fp)
    train_log_name = os.path.join(model_dir, "train.log")
    with open(train_log_name, 'w'):
        pass

    n_class = dataset.object_cfg.n_class
    n_epoch = config_dict['train_cfg']['n_epoch']
    batch_size = config_dict['train_cfg']['batch_size']
    lr = config_dict['train_cfg']['lr']
    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None

    print("Building dataloader ... (Mode: %s)" % ("save_memory" if args.save_memory else "normal"))


    crdata_train = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train', noise_channel=args.use_noise_channel)
    seq_names = crdata_train.seq_names
    index_mapping = crdata_train.index_mapping
    dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

    epoch_size = len(crdata_train) // batch_size


    if args.use_noise_channel:
        n_class_train = n_class + 1
    else:
        n_class_train = n_class

    print("Building model ... (%s)" % model_cfg)
    if model_cfg['model'] == 'T_RODNet':
        Cuda = True
        dice_loss = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        NUM_CLASSES = 3
        inputs_size = [128, 128, 2]
        net = T_RODNet(num_classes=NUM_CLASSES, embed_dim=64,win_size=4)
        # net = RODNet(in_channels=2, n_class=n_class_train).cuda()
        # net = Swin_RODNet(num_classes=NUM_CLASSES, embed_dim=64, win_size=4)

        checkpoint_path = r''
        if checkpoint_path != '':
            print('Load weights......')
            model_dict = net.state_dict()
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            print('Finished!')
        if Cuda:
            # net = torch.nn.DataParallel(net)
            # cudnn.benchmark = True
            net = net.cuda()
        criterion = nn.MSELoss()
    elif model_cfg['type'] == 'CDC':
        net = RODNet(in_channels=2, n_class=n_class_train).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HG':
        net = RODNet(in_channels=2, n_class=n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwI':
        net = RODNet(in_channels=2, n_class=n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'CDCv2':
        in_chirps = len(radar_configs['chirp_ids'])
        net = RODNet(in_channels=in_chirps, n_class=n_class_train,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGv2':
        in_chirps = len(radar_configs['chirp_ids'])
        net = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwIv2':
        in_chirps = len(radar_configs['chirp_ids'])
        net = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    else:
        raise TypeError
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    # scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch)
    iter_count = 0
    loss_ave = 0

    if cp_path is not None:
        checkpoint = torch.load(cp_path)
        if 'optimizer_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            iter_start = checkpoint['iter'] + 1
            loss_cp = checkpoint['loss']
            if 'iter_count' in checkpoint:
                iter_count = checkpoint['iter_count']
            if 'loss_ave' in checkpoint:
                loss_ave = checkpoint['loss_ave']
        else:
            net.load_state_dict(checkpoint)

    # print training configurations
    print("Model name: %s" % model_name)
    print("Number of sequences to train: %d" % crdata_train.n_seq)
    print("Training dataset length: %d" % len(crdata_train))
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size))

    loss_history = LossHistory("logs/", model_cfg['model'])
    start_time = time.time()
    for epoch in range(epoch_start, n_epoch):
        total_loss = 0
        tic_load = time.time()
        for iter, data_dict in enumerate(dataloader):
            image_paths = data_dict['image_paths']
            data = data_dict['radar_data']
            confmap_gt = data_dict['anno']['confmaps']

            data = torch.FloatTensor(data).to(device)

            realconfmaps = confmap_gt
            confmap_gt = torch.DoubleTensor(realconfmaps).to(device)

            if not data_dict['status']:
                # in case load npy fail
                print("Warning: Loading NPY data failed! Skip this iteration")
                tic_load = time.time()
                continue

            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients`

            confmap_preds_out = net(data)

            loss_confmap = 0
            loss_confmap = criterion(confmap_preds_out, confmap_gt.float())

            loss_confmap.backward()
            total_loss += loss_confmap.item()

            optimizer.step()

            if iter % config_dict['train_cfg']['log_step'] == 0:
                # print statistics

                # 学习率
                Nowlr = optimizer.state_dict()['param_groups'][0]['lr']
                # 计算时间
                seconds = (time.time() - start_time)
                day = seconds / 86400
                h = seconds % 86400 / 3600
                m = seconds % 86400 % 3600 / 60
                s = seconds % 86400 % 3600 % 60

                print(
                    'epoch %2d, iter %4d: loss: %.8f | load time: %.4f | backward time: %.4f | lr: %.8f | spend time:%2d天%2d时%2d分%2d秒' %
                    (epoch + 1, iter + 1, total_loss / (iter + 1), tic - tic_load, time.time() - tic, Nowlr, day, h, m,
                     s))
                with open(train_log_name, 'a+') as f_log:
                    f_log.write(
                        'epoch %2d, iter %4d: loss: %.8f | load time: %.4f | backward time: %.4f | lr: %.8f  | spend time:%2d天%2d时%2d分%2d秒\n' %
                        (epoch + 1, iter + 1, total_loss / (iter + 1), tic - tic_load, time.time() - tic, Nowlr, day, h,
                         m, s))

                writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)

                confmap_pred = confmap_preds_out.cpu().detach().numpy()

                data = torch.Tensor.cpu(data)
                chirp_amp_curr = chirp_amp(data[0, :, 0, :, :].cpu().detach().numpy(), radar_configs['data_type'])

                # draw train images
                fig_name = os.path.join(train_viz_path, '%03d_%010d_%06d.png' % (epoch + 1, iter_count, iter + 1))
                img_path = image_paths[0][0]

                visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                    confmap_pred[0, :n_class, 0, :, :],
                                    realconfmaps[0, :n_class, 0, :, :], model_cfg)

            if (iter + 1) % config_dict['train_cfg']['save_step'] == 0:
                # #validate current model
                # print("validing current model ...")
                # validate(confmap_preds_out[0])

                # save current model
                print("saving current model ...")
                status_dict = {
                    'model_name': model_name,
                    'epoch': epoch,
                    'iter': iter,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_confmap,
                    'iter_count': iter_count,
                }
                save_model_path = '%s/epoch_%02d_iter_%010d.pkl' % (model_dir, epoch + 1, iter_count + 1)
                torch.save(status_dict, save_model_path)

            iter_count += 1
            tic_load = time.time()

        loss_history.append_loss(total_loss / (epoch_size + 1))
        # save current model
        print("saving current epoch model ...")
        status_dict = {
            'model_name': model_name,
            'epoch': epoch,
            'iter': iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_confmap,
            'iter_count': iter_count,
        }
        save_model_path = '%s/epoch_%02d_final.pth' % (model_dir, epoch + 1)
        torch.save(net.state_dict(), save_model_path)

        scheduler.step()

    print('Training Finished.')
