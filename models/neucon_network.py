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


class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # import ipdb; ipdb.set_trace()
        if self.cfg.AGG_3DV=='sim':
            ch_in = [1 * alpha + 1, 96 + 1 * alpha + 2 + 1, 48 + 1* alpha + 2 + 1]
            channels = [96, 48, 24]
        elif self.cfg.AGG_3DV=='sim1/4':
            channels = [24, 12, 6]
            ch_in = [1 * alpha + 1, channels[0] + 1 * alpha + 2 + 1, channels[1] + 1* alpha + 2 + 1]
        elif self.cfg.AGG_3DV=='sim1/8':
            channels = [12, 6, 3]
            ch_in = [1 * alpha + 1, channels[0] + 1 * alpha + 2 + 1, channels[1] + 1* alpha + 2 + 1]
        else:
            ch_in = [80 * alpha + 1, 96 + 40 * alpha + 2 + 1, 48 + 24 * alpha + 2 + 1]#, 24 + 24 + 2 + 1]
            channels = [96, 48, 24]#out channel
        if cfg.CONCAT_DEPTH:
            ch_in = [ch+1 for ch in ch_in]
        elif cfg.CAT_PDEPTH:
            ch_in = [ch+1 if i>0 else ch for i, ch in enumerate(ch_in)]
        # print('************', ch_in)
        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(num_classes=1, in_channels=ch_in[i],
                       pres=1,
                       out_channel = channels[i], #cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            self.occ_preds.append(nn.Linear(channels[i], 1))

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
        ###xAdd
        if self.cfg.PROJ_WDEPTH or self.cfg.CONCAT_DEPTH:
            depth = inputs['depth'].detach()
            # print((depth==0).sum())
            depth = torch.nn.functional.interpolate(depth, scale_factor=1/8,recompute_scale_factor=True)#(bs,nviews,h/8,w/8)
        else:
            depth = None 
            # depth = inputs['depth'].detach()

        
        pre_depth = None
        # ----coarse to fine----
        for i in range(self.cfg.N_LAYER):#(0,1,2)
            interval = 2 ** (self.n_scales - i)#(4,2,1)
            scale = self.n_scales - i#(2,1,0)

            if i == 0:
                # ----generate new coords----
                coords = generate_grid(self.cfg.N_VOX, interval, device=features[0][0].device)[0]#N_VOX=96, interval=4 coords(24,24,24)
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)

            # ----back project----
            feats = torch.stack([feat[scale] for feat in features])
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            # import ipdb; ipdb.set_trace()            

            if pre_depth is not None:
                volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats,
                            KRcam, depth = pre_depth, 
                            sigma_factor = self.cfg.PROJ_WDEPTH, 
                            concat_depth= self.cfg.CAT_PDEPTH,
                            agg_v3d = self.cfg.AGG_3DV,
                            )
            else:
                volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats,
                                KRcam, depth = depth, 
                                sigma_factor = self.cfg.PROJ_WDEPTH, 
                                concat_depth= self.cfg.CONCAT_DEPTH,
                                agg_v3d = self.cfg.AGG_3DV,
                                )
            # print(self.cfg.AGG_3DV)
            # import ipdb; ipdb.set_trace()
            grid_mask = count > 1

            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            if not self.cfg.FUSION.FUSION_ON:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

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
            # print(feat.unique())
            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)
            # import ipdb; ipdb.set_trace()
            feat = self.sp_convs[i](point_feat)
            # import ipdb; ipdb.set_trace()
            # ----gru fusion----
            if self.cfg.FUSION.FUSION_ON:
                #up_coords will change:
                # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
                up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i)
                if self.cfg.FUSION.FULL:
                    grid_mask = torch.ones_like(feat[:, 0]).bool()
            # import ipdb; ipdb.set_trace()
            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)
            # print(feat.unique())
            # print(tsdf.unique())
            # print(occ.unique())

            # -------compute loss-------
            if tsdf_target is not None:
                occ_loss, tsdf_loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                         mask=grid_mask,
                                         pos_weight=self.cfg.POS_WEIGHT)
            else:
                occ_loss = torch.Tensor(np.array([0]))[0]
                tsdf_loss = torch.Tensor(np.array([0]))[0]

            loss_dict.update({f'occ_loss_{i}': occ_loss})
            loss_dict.update({f'tsdf_loss_{i}': tsdf_loss})


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

            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]
            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)

            if i == self.cfg.N_LAYER - 1:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf

            ##xAdded
            def cam_verify(intrinsics, extrinsics, scale):
                scale_intrinsics = intrinsics/4/2**scale
                scale_intrinsics[-1, -1] = 1
                proj_mat = torch.inverse(extrinsics)
                proj_mat[:3, :4] = scale_intrinsics @ proj_mat[:3, :4]
                return proj_mat

            if self.cfg.CAT_PDEPTH and i<self.cfg.N_LAYER: 
                # if self.cfg.CAT_PDEPTH and i==self.cfg.N_LAYER:
                if self.cfg.FUSION.FUSION_ON:
                    coords_updated = True
                else:
                    coords_updated = False  
                
                #Verify cam param.
                # for b in range(12):
                #     for c in range(9):
                #         proj_mat = cam_verify(inputs['intrinsics'][b,c], inputs['extrinsics'][b,c], scale)
                #         print((proj_mat-inputs['proj_matrices'][b, c, scale]).abs().mean())
                
                scale_intrinsics = inputs['intrinsics'] / 4 / 2 ** scale
                scale_intrinsics[:, :, -1, -1] = 1 #
                # proj_mat = torch.inverse(inputs['extrinsics'])# from (camera to world) to (world to camera)
                # proj_mat[:, :, :3, :4] = scale_intrinsics @ proj_mat[:, :, :3, :4]
                # diff = (proj_mat-inputs['proj_matrices'][:, :, scale]).abs()
                # print(diff.mean(), diff.max())
                # import ipdb; ipdb.set_trace()
                target_scale = scale-1 if scale>1 else scale
                pre_depth = self.proj_depth(pre_coords.detach(), pre_tsdf.detach(), inputs['vol_origin_partial'], 
                            scale_intrinsics, inputs['extrinsics'], i, coords_updated, world2cam=False, 
                            img_size=tuple(features[0][target_scale].shape[-2:]))#i:0,1,2, scale:2,1,0 
                
                # print('==========', (depth!=-1).sum())
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
        # loss = + 
        return loss_weight[0] * occ_loss, loss_weight[1] * tsdf_loss

