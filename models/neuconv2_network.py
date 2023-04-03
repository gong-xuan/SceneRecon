import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger

from models.modules import SPVCNN
from utils.utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project, back_project_visible
from ops.generate_grids import generate_grid, generate_rect_grid
from models.modules2D import TensoCNN

class NeuConV2Net(nn.Module):
    '''
    Coarse-to-fine network.
    '''
    def __init__(self, cfg):
        super(NeuConV2Net, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # import ipdb; ipdb.set_trace()
        if self.cfg.AGG_3DV=='sim':
            # ch_in = [1 * alpha + 1, 96 + 1 * alpha + 2 + 1, 48 + 1* alpha + 2 + 1]
            # channels = [96, 48, 24]
            ch_agg = 1 if self.cfg.AGG_3DV=='sim' else 24
            self.ch_in  = [ch_agg + cfg.CONCAT_IMZ]*3
            flatten_ch_in = [ch*24 for ch in self.ch_in] #24 h/w/d
            channels = [24*cfg.TENSO_CH]*3
            gru_ch = [cfg.TENSO_CH, cfg.TENSO_CH+2, cfg.TENSO_CH+2]
    
        else:
            import ipdb; ipdb.set_trace()
        
        # print('************', ch_in)
        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, gru_ch)
        # sparse conv
        self.convs= TensoCNN(in_channels=flatten_ch_in[0],
                       out_channel = channels[0], #cr=1 / 2 ** i,
                       dropout=self.cfg.SPARSEREG.DROPOUT)
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            # self.sp_convs.append(
            #     SPVCNN(num_classes=1, in_channels=ch_in[i],
            #            pres=1,
            #            out_channel = channels[i], #cr=1 / 2 ** i,
            #            vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
            #            dropout=self.cfg.SPARSEREG.DROPOUT)
            # )
            self.tsdf_preds.append(nn.Linear(gru_ch[i], 1))
            self.occ_preds.append(nn.Linear(gru_ch[i], 1))

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
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8(2**3)
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def process_3dfeat(self, inputs, feat2d):
        bs = feat2d.shape[1]
        # coords = generate_grid(self.cfg.N_VOX, interval=1)[0]#N_VOX=96, interval=4 coords(24,24,24)
        # up_coords = []
        # for b in range(bs):
        #     up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
        # up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
        KRcam = inputs['proj_matrices'][:, :, 0].permute(1, 0, 2, 3).contiguous()
        # volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feat2d, KRcam)
        
        down_ratio = 4
        feat = []
        grid_mask = []
        tv_loss = 0.
        for i in range(3):#(0,1,2)
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
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], voxel_size, feat2d,
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
                out_feat = out_feat.reshape(bs, self.cfg.TENSO_CH, *n_vox)
            elif i==1:
                out_feat = out_feat.reshape(bs, self.cfg.TENSO_CH, n_vox[1],  n_vox[0],  n_vox[2]).permute(0,1,3,2,4)
            elif i==2:
                out_feat = out_feat.reshape(bs, self.cfg.TENSO_CH, n_vox[2],  n_vox[0],  n_vox[1]).permute(0,1,3,4,2)
            ##############
            # if self.tvloss is not None:
            #     tv_loss += self.tvloss(out_feat, i)
            # import ipdb; ipdb.set_trace()
            # tvar(out_feat)
            count = count.reshape(bs, *n_vox)[:,None]
            out_feat = F.interpolate(out_feat, size=self.cfg.N_VOX, mode=self.cfg.UP)
            count = F.interpolate(count.float(), size=self.cfg.N_VOX, mode='nearest')
            feat.append(out_feat)
            grid_mask.append(count)
        feat = torch.stack(feat, dim=1).sum(dim=1)
        if self.cfg.NEW_GRID_MASK:
            grid_mask = back_project_visible(inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, self.cfg.N_VOX, KRcam, feat2d)
            grid_mask = (grid_mask>1).reshape(bs, *self.cfg.N_VOX) #(bs, D*H*W)
            # import ipdb; ipdb.set_trace()
            # grid_mask = grid_mask.view(-1)
        else:
            grid_mask = torch.cat(grid_mask, dim=1) #(bs, 3, H, W, D)
            grid_mask = (grid_mask>1).any(dim=1)#.view(-1)

        return feat, grid_mask

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
        pre_feat = None
        pre_coords = None
        loss_dict = {}
        #multi-scale process
        feat_fine2D= torch.stack([feat[0] for feat in features])
        nview, bs, _, _, _ = feat_fine2D.shape
        pool_kernel = [4,2,1]

        #project to 3D and conv
        feat_fine, grid_mask_fine = self.process_3dfeat(inputs, feat_fine2D) #(bs, ch, 96, 96, 96)

        # ----coarse to fine----
        for i in range(self.cfg.N_LAYER):#(0,1,2)
            # ----feat_3D----
            ks = pool_kernel[i]
            if ks!=1:
                volume = F.max_pool3d(feat_fine, ks)
                grid_mask = F.max_pool3d(grid_mask_fine.float(), ks)
                grid_mask = grid_mask.bool()
                # import ipdb;ipdb.set_trace()
            else:
                volume = feat_fine
                grid_mask = grid_mask_fine
            grid_mask = grid_mask.view(-1)
            ## 
            interval = 2 ** (self.n_scales - i)#(4,2,1)
            scale = self.n_scales - i#(2,1,0)
            voxel_size = tuple([n_vox//2**scale for n_vox in self.cfg.N_VOX])
            if i == 0:
                # ----generate new coords----
                coords = generate_grid(self.cfg.N_VOX, interval)[0]#N_VOX=96, interval=4 coords(24,24,24)
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)
            
            ## ----concat feature from last stage----
            volume = volume.permute(1,0,2,3,4).reshape(self.cfg.TENSO_CH,-1).permute(1,0)
            if i != 0:
                up_mask = F.interpolate(pre_mask[:,None].float(), size=voxel_size, mode='nearest')
                up_mask = (up_mask>0).view(-1) #no grad
                feat = torch.cat([volume[up_mask], up_feat], dim=1)
                grid_mask = grid_mask[up_mask]
            else:
                feat = volume

            # import ipdb; ipdb.set_trace()
            if not self.cfg.FUSION.FUSION_ON:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            # ----gru fusion----
            if self.cfg.FUSION.FUSION_ON:
                #up_coords will change:
                # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
                # import ipdb; ipdb.set_trace()
                up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i)
                if self.cfg.FUSION.FULL:
                    grid_mask = torch.ones_like(feat[:, 0]).bool()
            # 
            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)
            # print(feat.unique())
            # print(tsdf.unique())
            # print(occ.unique())

            # -------compute loss-------
            # import ipdb; ipdb.set_trace()
            if tsdf_target is not None:
                loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                         mask=grid_mask,
                                         pos_weight=self.cfg.POS_WEIGHT)
            else:
                loss = torch.Tensor(np.array([0]))[0]
            loss_dict.update({f'tsdf_occ_loss_{i}': loss})

            # ------define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i]
            occupancy[grid_mask == False] = False

            num = int(occupancy.sum().data.cpu())

            if num == 0:
                logger.warning('no valid points: scale {}'.format(i))
                return outputs, loss_dict

            # ------avoid out of memory: sample points if num of points is too large-----
            if self.training and num > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                choice = np.random.choice(num, num - self.cfg.TRAIN_NUM_SAMPLE[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy)
                occupancy[ind[choice]] = False

            pre_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict

            # pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]
            pre_feat = torch.cat([pre_tsdf, pre_occ], dim=1)
            #
            if i==0:
                pre_mask = occupancy.reshape(bs, *voxel_size)
            elif i!= self.cfg.N_LAYER - 1:
                if self.cfg.FUSION.FUSION_ON:
                    new_mask = torch.zeros((bs, *voxel_size)).bool().to(occupancy.device)
                    new_mask[up_coords[:,0],up_coords[:,1]//2**i, up_coords[:,2]//2**i, up_coords[:,3]//2**i] = occupancy
                else:
                    new_mask = up_mask.clone()
                    new_mask[up_mask] = occupancy
                # import ipdb; ipdb.set_trace()
                pre_mask = new_mask.reshape(bs, *voxel_size)
            
            if i == self.cfg.N_LAYER - 1:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf

            
        return outputs, loss_dict
    

    def proj_depth(self, coords, tsdf, origin, cam_intrinsics, cam_extrinsics, scale, coords_updated, world2cam, img_size):#3D to 2D
        """
        REFER TO: back_project
        3D: coords*voxelsize+origin
        tsdf->volume->mesh->depth
        """
        from utils import sparse_to_dense_channel
        from skimage import measure
        # import trimesh
        from tools.render_p3d import DynamicRenderer_Pytorch3D

        interval = 2 ** (self.n_scales - scale)
        # dim = (torch.Tensor(self.cfg.N_VOX).cuda() // interval).int() #for cuda10
        # dim = torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), interval, rounding_mode='trunc').int() #floor#for cuda11
        dim = torch.true_divide(torch.Tensor(self.cfg.N_VOX).cuda(), interval).trunc().int()
        dim_list = dim.data.cpu().numpy().tolist()
        #
        if not coords_updated:#tsdf truncated by 1
            # tsdf[tsdf.abs()>1] = 1
            invalid_tsdf = 1
            # import ipdb; ipdb.set_trace()
            trun_mask = (tsdf.abs()<1).squeeze()
            tsdf = tsdf[trun_mask]
            coords = coords[trun_mask]
        else:
            invalid_tsdf = 1
        #
        device = coords.device
        verts_list = []
        faces_list = []
        for i in range(origin.shape[0]):#batchsize
            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            if len(batch_ind) == 0:
                continue
            
            coords_b = coords[batch_ind, 1:].long() // interval
            tsdf_b = tsdf[batch_ind]
            origin_b = origin[i]
            
            tsdf_vol = sparse_to_dense_channel((coords_b).long().cpu(), 
                        tsdf_b.cpu(), dim_list, c=1,
                        default_val=invalid_tsdf, device='cpu') 
            #####NO gradients
            verts, faces, _, _ = measure.marching_cubes(tsdf_vol.squeeze(dim=-1).numpy(), level=None)#level=0
            verts = verts * self.cfg.VOXEL_SIZE + origin_b[None].cpu().numpy() # voxel grid coordinates to world coordinates
            # mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
            verts_list.append(torch.tensor(verts.copy()).to(device))
            faces_list.append(torch.tensor(faces.copy()).to(device))
            #####
        depth = DynamicRenderer_Pytorch3D(verts_list, faces_list, cam_intrinsics, cam_extrinsics,
            image_size = img_size, world2cam = world2cam, device=device)
        # import ipdb; ipdb.set_trace()
        depth[depth<0] = 0 #invalid value is 0
        return depth

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

