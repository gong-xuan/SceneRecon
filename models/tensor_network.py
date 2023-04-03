import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# from models.modules import SPVCNN
from models.modules2D import TensoCNN, SimpleTensoCNN, MsTensoCNN
from utils.utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project, back_project_visible
from ops.generate_grids import generate_grid, generate_rect_grid
from ops.loss import TVLoss

class TensorConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg, mode):
        super(TensorConNet, self).__init__()
        self.cfg = cfg
        self.n_sub = 3

        # alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])#1
        # import ipdb; ipdb.set_trace()
        ch_agg = 1 if self.cfg.AGG_3DV=='sim' else 24
        self.ch_in  = [ch_agg + cfg.CONCAT_IMZ]*3
        flatten_ch_in = [ch*24 for ch in self.ch_in] #24 h/w/d
        channels = [24*cfg.TENSO_CH]*3
        
        ch_multiplier = 1
        if cfg.TENSO=='smpl':
            conv2D = SimpleTensoCNN 
        elif cfg.TENSO=='ms':
            conv2D = MsTensoCNN 
            ch_multiplier = 3
        else:
            conv2D = TensoCNN
        self.convs= conv2D(in_channels=flatten_ch_in[0],
                       out_channel = channels[0], #cr=1 / 2 ** i,
                       dropout=self.cfg.SPARSEREG.DROPOUT)
        self.ch_2d = ch_multiplier*cfg.TENSO_CH
        if cfg.FUSE=='concat':
            self.ch_3d = 3*self.ch_2d
        else:
            self.ch_3d = self.ch_2d
        
        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, [self.ch_3d]*3) #TODO
        # MLPs that predict tsdf and occupancy.
        if cfg.MS_LOSS:
            self.tsdf_preds, self.occ_preds = nn.ModuleList(), nn.ModuleList()
            # inc = cfg.TENSO_CH*ch_multiplier
            # outc = 1
            for _ in range(3): 
                self.tsdf_preds.append(nn.Linear(self.ch_3d, 1))
                self.occ_preds.append(nn.Linear(self.ch_3d, 1))
        else:
            self.tsdf_preds = nn.Linear(self.ch_3d, 1)
            self.occ_preds = nn.Linear(self.ch_3d, 1)
        #
        # import ipdb; ipdb.set_trace()
        self.sparse = 'sparse' in mode
        self.weighted = 'weight' in mode
        if self.weighted or self.sparse:
            self.thresh = float(mode[-1])
        # import ipdb; ipdb.set_trace()
        if self.cfg.TV_LOSSW>0:
            self.tvloss = TVLoss(TVLoss_weight=self.cfg.TV_LOSSW)
        else:
            self.tvloss = None
        if self.cfg.TVC_LOSSW>0:
            self.tvcloss = TVLoss(TVLoss_weight=self.cfg.TVC_LOSSW)
        else:
            self.tvcloss = None
        if self.cfg.TVZ_LOSSW>0:
            self.tvzloss = TVLoss(TVLoss_weight=self.cfg.TVZ_LOSSW)
        else:
            self.tvzloss = None
        
    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            if coords is not None:
                coords_down = coords.detach().clone().long()
                # 2 ** scale == interval
                coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
                tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
                occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target


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
        scale = 0  
        feats = torch.stack([feat[scale] for feat in features])
        ##Remove??
        if self.cfg.COARSE_2D:
            NV, BS, _, H, W = feats.shape
            for s in [1,2]:
                ft = torch.stack([feat[s] for feat in features])
                ft = F.interpolate(ft.reshape(NV*BS,*ft.shape[2:]), size=(H,W), mode='nearest').reshape(NV,BS,ft.shape[2],H,W)
                feats = torch.cat([feats,ft], dim=2)
            # import ipdb; ipdb.set_trace()
        KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
        down_ratio = 4
        feat = []
        grid_mask = []

        tv_loss = 0.
        for i in range(self.n_sub):#(0,1,2)
            interval = [1 for _ in range(3)]
            interval[i] = down_ratio*interval[i]
            voxel_size = [self.cfg.VOXEL_SIZE * inter for inter in interval]
            #
            coords = generate_rect_grid(self.cfg.N_VOX, interval)[0]#[24, 96, 96]
            up_coords = []
            for b in range(bs):
                up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
            up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()

            # ----back project----
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], voxel_size, feats,
                                KRcam, 
                                concat_imz=self.cfg.CONCAT_IMZ,
                                depth = None, 
                                sigma_factor = self.cfg.PROJ_WDEPTH, 
                                concat_depth= self.cfg.CONCAT_DEPTH,
                                agg_v3d = self.cfg.AGG_3DV,
                                )
            # print(self.cfg.AGG_3DV)
            # valid_mask = count > 1
            # exist_mask = count>0

            n_vox = [n for n in self.cfg.N_VOX]
            n_vox[i] = n_vox[i]//down_ratio
            ch_in = self.ch_in[i]
            if i==0:
                in_feat = volume.reshape(bs, *n_vox, ch_in).permute(0,4,1,2,3).reshape(bs, ch_in*n_vox[0],*n_vox[1:])
            elif i==1:
                in_feat = volume.reshape(bs, *n_vox, ch_in).permute(0,4,2,1,3).reshape(bs, ch_in*n_vox[1],n_vox[0],n_vox[2])
            elif i==2:
                in_feat = volume.reshape(bs, *n_vox, ch_in).permute(0,4,3,1,2).reshape(bs, ch_in*n_vox[2],n_vox[0],n_vox[1])

            out_feat = self.convs(in_feat)
            if i==0:
                out_feat = out_feat.reshape(bs, self.ch_2d, *n_vox)
            elif i==1:
                out_feat = out_feat.reshape(bs, self.ch_2d, n_vox[1],  n_vox[0],  n_vox[2]).permute(0,1,3,2,4)
            elif i==2:
                out_feat = out_feat.reshape(bs, self.ch_2d, n_vox[2],  n_vox[0],  n_vox[1]).permute(0,1,3,4,2)
            ##############
            if self.tvloss is not None:
                tv_loss += self.tvloss(out_feat, i)
            # import ipdb; ipdb.set_trace()
            # tvar(out_feat)
            count = count.reshape(bs, *n_vox)[:,None]
            # in_feat all zero where count===0

            out_feat = F.interpolate(out_feat, size=self.cfg.N_VOX, mode=self.cfg.UP)
            count = F.interpolate(count.float(), size=self.cfg.N_VOX, mode='nearest')
            feat.append(out_feat)
            grid_mask.append(count)
        
        if self.cfg.NEW_GRID_MASK:
            grid_mask = back_project_visible(inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, self.cfg.N_VOX, KRcam, feats)
            grid_mask = (grid_mask>1) #(bs, D*H*W)
            # import ipdb; ipdb.set_trace()
        else:
            grid_mask = torch.cat(grid_mask, dim=1) #(bs, 3, H, W, D)
        # print(feat.unique())
        # print(tsdf.unique())
        # print(occ.unique())
        coords = generate_grid(self.cfg.N_VOX, 1)[0]
        up_coords = []
        for b in range(bs):
            up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
        up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
        

        # import ipdb; ipdb.set_trace()
        if self.sparse:
            feat = torch.cat(feat, dim=1).permute(0,2,3,4,1)
            if self.thresh==1:
                mask = (grid_mask>1).any(dim=1)#TODO any
                feat = feat[mask]
                up_coords = up_coords[mask.view(-1)]
                grid_mask = torch.ones_like(feat[:, 0]).bool()
            else:
                mask = (grid_mask>0).any(dim=1)
                feat = feat[mask]
                up_coords = up_coords[mask.view(-1)]
                grid_mask = (grid_mask>1).any(dim=1).view(-1)
                grid_mask = grid_mask[mask.view(-1)]
                # import ipdb; ipdb.set_trace()
        elif self.weighted:
            feat = torch.stack(feat, dim=1)#()
            mask0 = (grid_mask>self.thresh) #0 or 1
            mask_feat = torch.stack([mask0 for _ in range(self.cfg.TENSO_CH)], dim=2)
            feat = (feat*mask_feat).reshape(bs, 3*self.cfg.TENSO_CH, *self.cfg.N_VOX)
            feat = feat.permute(0,2,3,4,1).reshape(-1, self.cfg.TENSO_CH*3)
            grid_mask = (grid_mask>1).any(dim=1).view(-1)
            # import ipdb; ipdb.set_trace()
        else:
            # import ipdb; ipdb.set_trace()
            if self.cfg.FUSE=='concat':
                feat = torch.cat(feat, dim=1)
            elif self.cfg.FUSE=='plus':
                feat = torch.stack(feat, dim=1).sum(dim=1)
            elif self.cfg.FUSE=='mul':
                feat = torch.stack(feat, dim=1)
                # import ipdb; ipdb.set_trace()
                feat = feat[:,0]*feat[:,1]*feat[:,2]
            if self.tvcloss is not None:
                tv_loss +=self.tvcloss(feat, 0)
                tv_loss +=self.tvcloss(feat, 1)
                tv_loss +=self.tvcloss(feat, 2)
            if self.tvzloss is not None:
                tv_loss +=self.tvzloss(feat, 2)
            
            feat = feat.permute(0,2,3,4,1).reshape(-1, feat.shape[1])
            
            if self.cfg.NEW_GRID_MASK:
                grid_mask = grid_mask.view(-1)
            else:
                grid_mask = (grid_mask>1).any(dim=1).view(-1)
        
        
        # ----gru fusion---- #TODO
        if self.cfg.FUSION.FUSION_ON:
            """
            num = up_coords.shape[0]
            THRESH  = 100000
            if self.training and num > THRESH * bs: #96*96*96=884736
                choice = np.random.choice(num, THRESH * bs, replace=False)
                # ind = torch.range(0, num)
                up_coords = up_coords[choice]
                print('=======================', num, num//bs,  'TO', THRESH * bs)
                # import ipdb; ipdb.set_trace()
                # up_coords[ind[choice]] = False
            """
            # import ipdb; ipdb.set_trace()
            #up_coords will change:
            # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
            up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, 2)
            if self.cfg.FUSION.FULL:
                grid_mask = torch.ones_like(feat[:, 0]).bool()
        else:
            coords_target = up_coords if self.sparse else None
            tsdf_target, occ_target = self.get_target(coords_target, inputs, scale)
        ###Loss##
        if self.cfg.MS_LOSS:
            tsdf, occupancy, loss_dict = self.feat_to_msloss(feat, tsdf_target, occ_target, grid_mask, bs)
        else:
            tsdf, occupancy, loss_dict = self.feat_to_loss(feat, tsdf_target, occ_target, grid_mask)
        
        if (self.tvloss is not None) or (self.tvcloss is not None) or (self.tvzloss is not None):
            loss_dict.update({f'tv_loss': tv_loss})
        
        # import ipdb; ipdb.set_trace()
        if tsdf is not None and occupancy is not None:
            coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: batch {}'.format(b))
                    return outputs, loss_dict
            outputs['coords'] = coords
            outputs['tsdf'] = tsdf[occupancy]
        return outputs, loss_dict

    def feat_to_msloss(self, feat_fine, tsdf_target_fine, occ_target_fine, grid_mask_fine, batchsize):
        """
        feat_fine: (bs*96*96*96, ch)
        tsdf_target_fine: (bs,96,96,96) or (bs*96*96*96,1)
        occ_target_fine: (bs,96,96,96) or (bs*96*96*96,1)
        grid_mask_fine: (bs*96*96*96)
        """
        loss_dict = {}
        bvol_size = [batchsize]+self.cfg.N_VOX
        pool_kernel = [4,2,1]
        feat_ch = feat_fine.shape[-1]
        if feat_fine.shape[0]!=batchsize*96*96*96:
            import ipdb; ipdb.set_trace()
        for i,ks in enumerate(pool_kernel):
            if ks>1:
                # import ipdb; ipdb.set_trace()
                # print(feat_fine.shape)
                feat = F.max_pool3d(feat_fine.permute(1,0).reshape(feat_ch, *bvol_size), ks).reshape(feat_ch, -1).permute(1,0) 
                grid_mask = F.max_pool3d(grid_mask_fine.reshape(*bvol_size)[:,None].float(), ks).view(-1).bool()
                tsdf_target = F.max_pool3d(tsdf_target_fine.reshape(*bvol_size)[:,None], ks).view(-1) 
                occ_target = F.max_pool3d(occ_target_fine.reshape(*bvol_size)[:,None].float(), ks).view(-1).bool() 
                # 
            else:
                feat = feat_fine
                tsdf_target = tsdf_target_fine
                occ_target = occ_target_fine
                grid_mask = grid_mask_fine
            
            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)
            # -------compute loss-------
            assert tsdf_target is not None
            occ_loss, tsdf_loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                    mask=grid_mask, pos_weight=self.cfg.POS_WEIGHT) #pos_weight=1.25
           
            loss_dict.update({f'occ_loss_{i}': occ_loss})
            loss_dict.update({f'tsdf_loss_{i}': tsdf_loss})

        occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[2]
        occupancy[grid_mask == False] = False
        num = int(occupancy.sum().data.cpu())
        if num == 0:
            logger.warning('no valid points')
            return None, None, loss_dict

        return tsdf, occupancy, loss_dict

    def feat_to_loss(self, feat, tsdf_target, occ_target, grid_mask):
        loss_dict = {}
        tsdf = self.tsdf_preds(feat)
        occ = self.occ_preds(feat)

        # -------compute loss-------
        if tsdf_target is not None:
            occ_loss, tsdf_loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                        mask=grid_mask,
                                        pos_weight=self.cfg.POS_WEIGHT) #pos_weight=1.25
        else:
            occ_loss = torch.Tensor(np.array([0]))[0]
            tsdf_loss = torch.Tensor(np.array([0]))[0]

        loss_dict.update({f'occ_loss': occ_loss})
        loss_dict.update({f'tsdf_loss': tsdf_loss})
        
        
        occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[2]
        occupancy[grid_mask == False] = False
        num = int(occupancy.sum().data.cpu())
        if num == 0:
            logger.warning('no valid points')
            return None, None, loss_dict
        return tsdf, occupancy, loss_dict


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
        return loss_weight[0] * occ_loss, loss_weight[1] * tsdf_loss

