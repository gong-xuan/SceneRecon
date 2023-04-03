import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger

from models.modules import SPVCNN
from utils.utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid

from models.neucon_network import NeuConNet

class RaftNeuConNet(NeuConNet):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg, mode):
        super(NeuConNet, self).__init__()
        self.cfg = cfg

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # import ipdb; ipdb.set_trace()
        
        ch_in = 1 * alpha + 1
        channels = 96
        
        # print('************', ch_in)

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, [channels])
        # sparse conv
        self.sp_convs = SPVCNN(num_classes=1, in_channels=ch_in,
                       pres=1,
                       out_channel = channels, #cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE,
                       dropout=self.cfg.SPARSEREG.DROPOUT)
        # MLPs that predict tsdf and occupancy. 
        self.tsdf_preds = nn.Linear(channels, 1)
        self.occ_preds = nn.Linear(channels, 1)
        #
        mode = mode.split('-')
        self.n_gru = int(mode[1])
        self.mode = mode[0]

    def forward(self, features, inputs, outputs):
        '''
        :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
        :param inputs: meta data from dataloader
        :param outputs: {}
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
        }
        :return: loss_dict: dict: {x
            'tsdf_occ_loss_X':         (Tensor), multi level loss
        }
        '''
        bs = features[0][0].shape[0]
        pre_feat = None
        pre_coords = None
        loss_dict = {}

        # 
        # for i in range(self.cfg.N_LAYER):#(0,1,2)
        if True:
            interval = 1
            scale = 0

            # ----generate new coords----
            coords = generate_grid(self.cfg.N_VOX, interval)[0]#N_VOX=96, interval=1
            up_coords = []
            for b in range(bs):
                up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
            up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            
            # ----back project----
            feats = torch.stack([feat[scale] for feat in features])
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            # import ipdb; ipdb.set_trace()
            
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats,
                                KRcam, depth = None, 
                                sigma_factor = 1, 
                                concat_depth= 0,
                                agg_v3d = 'sim',
                                )
            # print(self.cfg.AGG_3DV)
            # import ipdb; ipdb.set_trace()
            grid_mask = count > 1
            feat = volume
            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            for n in range(self.n_gru):
                # ------avoid out of memory: sample points if num of points is too large-----
                if self.mode=='hierandom':
                    # train_max_sample  = self.cfg.TRAIN_NUM_SAMPLE[-1]
                    train_max_sample = 96*96*96//8#**(n+1)
                    # train_max_sample = 96*96*96//16**(n+2)

                    num = r_coords.shape[0] if n==0 else pre_coords.shape[0]
                    if self.training and num >  train_max_sample* bs:
                        # print('Prune', n)
                        choice = np.random.choice(num, train_max_sample * bs,
                                                replace=False)
                        choice.sort()
                        # import ipdb; ipdb.set_trace()
                        if n==0:
                            cur_r_coords = r_coords[choice]
                            cur_up_coords = up_coords[choice]
                            cur_grid_mask = grid_mask[choice]
                        else:
                            cur_r_coords  = cur_r_coords[occupancy][choice]
                            cur_up_coords = pre_coords[choice]
                            cur_grid_mask = cur_grid_mask[occupancy][choice]
                    else:
                        cur_r_coords = r_coords
                        cur_up_coords = up_coords
                        cur_grid_mask = grid_mask
                    # ind = torch.nonzero(occupancy)
                    # occupancy[ind[choice]] = False
                elif self.mode=='random':
                    train_max_sample = 96*96*96/2
                    num = r_coords.shape[0]
                    if self.training:
                        choice = np.random.choice(num, num - train_max_sample * bs,
                                                replace=False)
                        cur_r_coords = r_coords[choice]
                        cur_up_coords = up_coords[choice]
                elif self.mode=='hier':
                    if self.training:
                        if n==0:
                            cur_r_coords = r_coords
                            cur_up_coords = up_coords
                            cur_grid_mask = grid_mask

                        else:
                            train_max_sample = [65536, 65536][n-1]
                            num = pre_coords.shape[0]
                            #not valid when fusion_on=True
                            if num >  train_max_sample* bs:
                                choice = np.random.choice(num, num - train_max_sample * bs,
                                                    replace=False)
                                cur_r_coords = cur_r_coords[occupancy][choice]
                                cur_up_coords = pre_coords[choice]
                                cur_grid_mask = cur_grid_mask[occupancy][choice]
                            else:
                                cur_r_coords = cur_r_coords[occupancy]
                                cur_up_coords = pre_coords
                                cur_grid_mask = cur_grid_mask[occupancy]
                # ----sparse conv 3d backbone----
                # print(cur_r_coords.shape)
                # print(feat.shape, cur_r_coords.shape)
                # print(cur_r_coords.max(), cur_r_coords.min())
                # import ipdb; ipdb.set_trace()
                point_feat = PointTensor(feat, cur_r_coords)
                cur_feat = self.sp_convs(point_feat)
                
                # ----gru fusion----
                if self.cfg.FUSION.FUSION_ON:
                    #up_coords will change:
                    # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
                    cur_up_coords, cur_feat, tsdf_target, occ_target = self.gru_fusion(cur_up_coords, cur_feat, inputs, 0)
                    if self.cfg.FUSION.FULL:
                        cur_grid_mask = torch.ones_like(cur_feat[:, 0]).bool()
                else:
                    tsdf_target, occ_target = self.get_target(cur_up_coords, inputs, scale)
                # import ipdb; ipdb.set_trace()
                tsdf = self.tsdf_preds(cur_feat)
                occ = self.occ_preds(cur_feat)

                # -------compute loss-------
                if tsdf_target is not None:
                    loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                            mask=cur_grid_mask,
                                            pos_weight=self.cfg.POS_WEIGHT)
                else:
                    loss = torch.Tensor(np.array([0]))[0]
                loss_dict.update({f'tsdf_occ_loss': loss})

                # ------define the sparsity for the next stage-----
                # import ipdb; ipdb.set_trace()
                occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[0]
                occupancy[cur_grid_mask == False] = False

                num = int(occupancy.sum().data.cpu())

                if num == 0:
                    logger.warning('no valid points')
                    return outputs, loss_dict
                
                pre_coords = cur_up_coords[occupancy]
                for b in range(bs):
                    batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                    if len(batch_ind) == 0:
                        logger.warning('no valid points: batch {}'.format(b))
                        return outputs, loss_dict

                # pre_feat = feat[occupancy]
                # pre_tsdf = tsdf[occupancy]
                # pre_occ = occ[occupancy]
                # pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)
            
            outputs['coords'] = pre_coords
            outputs['tsdf'] = tsdf[occupancy]

        return outputs, loss_dict
    

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        '''
        :param tsdf: (Tensor), predicted tsdf, (N, 1)
        :param occ: (Tensor), predicted occupancy, (N, 1)
        :param tsdf_target: (Tensor),ground truth tsdf, (N, 1)
        :param occ_target: (Tensor), ground truth occupancy, (N, 1)
        :param loss_weight: (Tuple)
        :param mask: (Tensor), mask voxels which cannot be seen by all views
        :param pos_weight: (float)
        :return: loss: (Tensor)
        '''
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))
        # print('occ', occ_loss.item(), 'tsdf', tsdf_loss.item())
        if torch.isnan(occ_loss):
            import ipdb; ipdb.set_trace()
            
        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss

