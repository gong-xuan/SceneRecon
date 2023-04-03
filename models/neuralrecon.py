import torch
import torch.nn as nn

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from .raft_network import RaftNeuConNet
from .tensor_network import TensorConNet
from .neuconv1_network import NeuConV1Net
from .neuconv2_network import NeuConV2Net
from .neuconv3_network import NeuConV3Net
from .neuconv4_network import NeuConV4Net


from loguru import logger


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg, mode=''):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
            
        if mode.startswith('raft'):
            self.neucon_net = RaftNeuConNet(cfg.MODEL, mode[3:])
        elif mode.startswith('tens'):
            self.neucon_net = TensorConNet(cfg.MODEL, mode.split('-')[1])
        elif mode.startswith('neuv1'):
            self.neucon_net = NeuConV1Net(cfg.MODEL)
        elif mode.startswith('neuv2'):
            self.neucon_net = NeuConV2Net(cfg.MODEL)
        elif mode.startswith('neuv3'):
            self.neucon_net = NeuConV3Net(cfg.MODEL)
        elif mode.startswith('neuv4'):
            self.neucon_net = NeuConV4Net(cfg.MODEL)
        else:
            self.neucon_net = NeuConNet(cfg.MODEL)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

        self.mode = mode

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''

        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]
        # import ipdb; ipdb.set_trace()
        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)
        

        # fuse to global volume.

        # import ipdb; ipdb.set_trace()
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)
        
        weighted_loss = 0
        # import ipdb; ipdb.set_trace()
        if self.mode.startswith('tens'):
            for (k,v) in loss_dict.items():
                weighted_loss += v
        else:
            for i, (k, v) in enumerate(loss_dict.items()):
                weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        

        return outputs, loss_dict
