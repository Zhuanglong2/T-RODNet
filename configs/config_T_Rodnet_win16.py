dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="/media/myssd/Datasets/RADAR/RADAR/mnt/disk1/CRUW/ROD2021",
    data_root="/media/myssd/Datasets/RADAR/RADAR/mnt/disk1/CRUW/ROD2021/sequences",
    anno_root=r"/media/myssd/Datasets/RADAR/RADAR/mnt/disk1/CRUW/ROD2021/annotations",
    anno_ext='.txt',
    train=dict(
        subdir='train',
        # seqs=[],  # can choose from the subdir folder
    ),
    valid=dict(
        subdir='valid',
        seqs=[],
    ),
    test=dict(
        subdir='test',
        #seqs=['2019_05_28_PCMS004'],  # can choose from the subdir folder
    ),
    demo=dict(
        subdir='demo',
        seqs=[],
    ),
)

model_cfg = dict(
    type='CDC',
    model='T_RODNet',
    name='T_RODNet',
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
    segment_thres = 0,
)

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 15,
        'cyclist': 20,
        'car': 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
        # 'van': 4,
        # 'truck': 5,
    }
)

train_cfg = dict(
    n_epoch=100,
    batch_size=4,
    lr=0.0001,
    lr_step=5,
    win_size=16,
    train_step=1,
    train_stride=4,
    log_step=100,
    save_step=100000,
)
test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)
