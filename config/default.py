from yacs.config import CfgNode as CN

_C = CN()

_C.MODE = 'train'
_C.DATASET = 'scannet'
_C.TSDF_ROOT = ''
_C.LOADCKPT = ''
_C.LOGDIR = './checkpoints'
_C.RESUME = True
_C.SUMMARY_FREQ = 20
_C.SEED = 1

_C.SAVE_INCREMENTAL = False
_C.VIS_INCREMENTAL = False
_C.REDUCE_GPU_MEM = False

_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False

# train
_C.TRAIN = CN()
_C.TRAIN.PATH = ''
_C.TRAIN.EPOCHS = 40
_C.TRAIN.LR = 0.001
_C.TRAIN.LREPOCHS = '12,24,36:2'
_C.TRAIN.WD = 0.0
_C.TRAIN.N_VIEWS = 5
_C.TRAIN.N_WORKERS = 8
_C.TRAIN.RANDOM_ROTATION_3D = True
_C.TRAIN.RANDOM_TRANSLATION_3D = True
_C.TRAIN.PAD_XY_3D = .1
_C.TRAIN.PAD_Z_3D = .025
_C.TRAIN.SAVE_FREQ = 1
_C.TRAIN.BATCH_SIZE = 1

# test
_C.TEST = CN()
_C.TEST.PATH = ''
_C.TEST.N_VIEWS = 5
_C.TEST.N_WORKERS = 4
_C.TEST.SAVE_SCENE_MESH = ''
_C.TEST.BATCH_SIZE = 1

# model
_C.MODEL = CN()
_C.MODEL.N_VOX = [128, 224, 192]
_C.MODEL.VOXEL_SIZE = 0.04
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.N_LAYER = 3

_C.MODEL.TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
_C.MODEL.TEST_NUM_SAMPLE = [32768, 131072]

_C.MODEL.LW = [1.0, 0.8, 0.64]

# TODO: images are currently loaded RGB, but the pretrained models expect BGR
_C.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_STD = [1., 1., 1.]
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.POS_WEIGHT = 1.0

_C.MODEL.BACKBONE2D = CN()
_C.MODEL.BACKBONE2D.ARC = 'fpn-mnas'

_C.MODEL.SPARSEREG = CN()
_C.MODEL.SPARSEREG.DROPOUT = False

_C.MODEL.FUSION = CN()
_C.MODEL.FUSION.FUSION_ON = False
_C.MODEL.FUSION.HIDDEN_DIM = 64
_C.MODEL.FUSION.AVERAGE = False
_C.MODEL.FUSION.FULL = False


def update_config(cfg, args):
    cfg.defrost()
    cfgfile = f'./config/{args.cfg}_pre.yaml' if args.pretrain else f'./config/{args.cfg}.yaml'
    cfg.merge_from_file(cfgfile)
    cfg.merge_from_list(args.opts)

    #
    cfg.defrost()
    
    cfg.LOCAL_RANK = args.local_rank
    cfg.MODEL.PROJ_WDEPTH = args.proj_wdepth
    cfg.MODEL.CONCAT_DEPTH = args.concat_depth
    cfg.MODEL.AGG_3DV = args.agg3dv
    cfg.MODEL.CAT_PDEPTH = args.catpdepth
    #for tenso_network only
    cfg.MODEL.TENSO = args.tenso
    cfg.MODEL.TENSO_CH = args.tsch
    cfg.MODEL.FUSE = args.fuse
    cfg.MODEL.UP = args.up
    cfg.MODEL.NEW_GRID_MASK = args.newgridmask
    cfg.MODEL.TV_LOSSW = args.tvloss
    cfg.MODEL.TVC_LOSSW = args.tvcloss
    cfg.MODEL.TVZ_LOSSW = args.tvzloss

    cfg.MODEL.CONCAT_IMZ = args.concatimz
    cfg.MODEL.COARSE_2D = args.coarse2d
    cfg.MODEL.MS_LOSS = args.msloss

    cfg.GT_TSDF = [float(t) for t in args.tsdf.split(',')]
    if args.batchsize:
        cfg.BATCH_SIZE = int(args.batchsize)
    if args.dataset:
        cfg.DATASET = args.dataset
        cfg.TEST.PATH = '/rawdata/xuangong/data/7scenes'
    # if args.proj_wdepth:
    #     cfg.LOGDIR = f'{cfg.LOGDIR}pwd'
    # cfg.LOGDIR = f'{cfg.LOGDIR}/{args.log}'
    cfg.freeze()

def check_config(cfg):
    pass
