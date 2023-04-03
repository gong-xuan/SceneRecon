from datasets.sampler import DistributedSampler
from torch.utils.data import DataLoader
from datasets.transforms import ResizeImage, ToTensor, RandomTransformSpace, IntrinsicsPoseToProjection, Compose
import torch
from loguru import logger
import importlib
# find the dataset definition by name, for example ScanNetDataset (scannet.py)

def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)

    if dataset_name == 'scannet':
        return getattr(module, "ScanNetDataset")
    elif dataset_name =='sevenscenes':
        return getattr(module, 'SevenSceneDataset')
    elif dataset_name == 'demo':
        return getattr(module, "DemoDataset")


def build_dataloader(cfg, mode, max_depth=3., vol_org_z=-0.2, val_subset=1):
    """
    mode: train, val, test
    """
    # Augmentation
    if mode == 'train':
        n_views = cfg.TRAIN.N_VIEWS
        random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
        random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
        paddingXY = cfg.TRAIN.PAD_XY_3D
        paddingZ = cfg.TRAIN.PAD_Z_3D
    else:
        n_views = cfg.TEST.N_VIEWS
        random_rotation = False
        random_translation = False
        paddingXY = 0
        paddingZ = 0
    # print('====maxdepth====', max_depth)
    transform = []
    transform += [ResizeImage((640, 480)), ##（W,H): same for sevenscenes
                ToTensor(),
                RandomTransformSpace(
                    cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                    paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS, max_depth=max_depth,
                    vol_org_z = vol_org_z, trunc_sdf=cfg.GT_TSDF),
                IntrinsicsPoseToProjection(n_views, 4),
                ]

    transforms = Compose(transform)

    # dataset, dataloader
    # if cfg.DATASET=='sevenscenes':
    #     from datasets.sevenscenes import SevenSceneDataset
    #     dataset = SevenSceneDataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
    # else:
    MVSDataset = find_dataset_def(cfg.DATASET)

    if mode == 'train':
        dataset = MVSDataset(cfg.TRAIN.PATH, "train", transforms, cfg.TSDF_ROOT, cfg.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
    elif mode == 'test':
        dataset = MVSDataset(cfg.TEST.PATH, "test", transforms, cfg.TSDF_ROOT, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, max_depth=max_depth)
    elif mode == 'val':
        dataset = MVSDataset(cfg.TEST.PATH, "val", transforms, cfg.TSDF_ROOT, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, subset=val_subset)
    #
    logger.info(f"Load {mode} dataset: {len(dataset)}")

    batch_size = cfg.TEST.BATCH_SIZE if mode=='test' else cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TEST.N_WORKERS if mode=='test' else cfg.TRAIN.N_WORKERS
    drop_last = False if mode=='test' else True

    if cfg.DISTRIBUTED:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    else:
        dataLoader = DataLoader(dataset, 
                                batch_size, 
                                shuffle=False, 
                                num_workers=num_workers,
                                drop_last=False)
    return dataLoader