from operator import concat
import torch
from torch.nn.functional import grid_sample
from ops.generate_grids import generate_grid
EPS=1e-5

def back_project(coords, origin, voxel_size, feats, KRcam, concat_imz = 1, depth=None, sigma_factor=1, concat_depth=0, agg_v3d=''):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume
    if depth is not None:
        sigma_factor=0, concat
        sigma_factor>0, masking

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    device = feats.device
    n_views, bs, c, h, w = feats.shape #feat channel 80+1
    if depth is not None and concat_depth:
        add_ch = 2
    elif concat_imz:
        add_ch = 1
    else:
        add_ch = 0
    if agg_v3d.startswith('sim'):
        c_featout = 1
    else:
        c_featout = c
    feature_volume_all = torch.zeros(coords.shape[0], c_featout + add_ch).to(device)
    count = torch.zeros(coords.shape[0]).to(device)

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]
        
        if not isinstance(voxel_size, list):
            grid_batch = coords_batch * voxel_size + origin_batch.float()
        else:
            grid_batch = coords_batch * torch.tensor(voxel_size)[None].to(device) + origin_batch.float()
        
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).to(device)], dim=1)#(nview, 4, nV)

        # Project grid, nV=24/48/96^3
        im_p = proj_batch @ rs_grid #(nview, 4, nV)
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]#im_z exits 0
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0) #inside image

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2) #(nview, 1, nV, 2) 
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)#(nview, c, 1, nV)
        # if torch.isnan(features).any().item():#im_z=0, im_grid=inf, features=nan
        #     import ipdb; ipdb.set_trace()

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()

        
        ##xAdd
        if depth is not None:
            h_depth, w_depth = depth.shape[2:]
            assert h_depth/h==w_depth/w #scale ratio: depth to feat
            #
            im_grid_fordepth = im_grid* h_depth/h
            #
            depth_batch = depth[batch][:,None] #(nview, 1, h0, w0)
            depth_volum = grid_sample(depth_batch, im_grid_fordepth, padding_mode='zeros', align_corners=True)#(nview, 1, 1, nV)
            depth_volum = depth_volum.squeeze()#(nview, nV)
            zero_mask = (depth_volum == 0) #depth=0, no clue
            delta_depth = (depth_volum - im_z).abs()/(EPS+depth_volum) 
            print((delta_depth[~zero_mask]<0.08).sum())
            # import ipdb; ipdb.set_trace()
            # for sigma_factor in [0.5, 1, 2,5]:
            if sigma_factor>0:
                sigma = sigma_factor*torch.ones_like(im_z)#TODO:grid_sample from 2D uncert 
                #count prob=1: 10->200, 100->1055 (24*24=576)
                prob = torch.exp(-0.5*((delta_depth/sigma)**2))#/(sigma*math.sqrt(2*math.pi))
                prob = prob*(~zero_mask)+torch.ones_like(zero_mask)*zero_mask#[0,1] #there may exist nan in prob
                prob[mask == False] = 0 
                # print(prob.unique())
                # import ipdb; ipdb.set_trace()
                if concat_depth:
                    features = torch.cat([features, prob[:,None]], dim=1)
                else:
                    # prob = 0.5+ prob/2 #[0.5,1]
                    # prob[prob<0.5] = 0.5
                    # prob[prob>1] = 1
                    features = prob[:,None].detach()*features
                # features = 0.5*torch.ones_like(features)*features
                #check hyperparam: sigma
                if False:
                    valid_prob = prob.clone()
                    valid_prob[zero_mask] = 0
                    # # for thresh in [0.2, 0.5, 0.8]:
                    for thresh in [0.8]:
                        print(f'Sigma:{sigma_factor}, >{thresh} prob:', (valid_prob>thresh).sum().item()/(valid_prob>0).sum().item())

            # import ipdb; ipdb.set_trace()
        if torch.isnan(features).any().item():
            import ipdb; ipdb.set_trace()
        # aggregate multi view #(nview, c, NV)
        in_scope_mask = mask.sum(dim=0)
        # invalid_mask = mask == 0
        # import ipdb; ipdb.set_trace()
        in_scope_mask[in_scope_mask == 0] = 1 ##features might be all 0
        in_scope_mask = in_scope_mask.unsqueeze(0) #(1, NV)
        if (in_scope_mask==0).any():
            import ipdb; ipdb.set_trace()
        features_mean = features.sum(dim=0)
        features_mean /= in_scope_mask #(c, NV)
        
        if agg_v3d=='var':
            feat_diff = features - features_mean[None] #(9,c,NV)
            # feat_diff [(mask==0)[:,None].expand(-1, c, -1) == False] = 0#mask:(9, NV)
            feat_diff [(mask==0)[:,None].expand(-1, c, -1)] = 0 #invalid->0, diff
            featvar = feat_diff.pow(2).sum(dim=0) #(c,NV)
            featvar /= in_scope_mask
            featvar = featvar.permute(1, 0).contiguous()# (NV, c)
            features = featvar
        elif agg_v3d.startswith('sim'):
            feat_view_correlation= torch.einsum('bij,bjk->bik', 
                    features.transpose(0,2).transpose(1,2), #(NV, 9, c)
                    features.transpose(0,2)) #(NV, c, 9)
            #remove effect of invalid view
            feat_view_correlation[(mask==0).transpose(0,1)[:,:,None].expand(-1,-1,n_views)] = 0 #(NV, 9, 9)
            valid_n = (feat_view_correlation!=0).sum(dim=1).sum(dim=1) #(NV)
            # import ipdb; ipdb.set_trace()

            features = feat_view_correlation.sum(dim=1).sum(dim=1)/(valid_n+1e-5)
            features = features[:,None] #(NV, 1)
            # 
        else:
            features_mean = features_mean.permute(1, 0).contiguous()# (NV, c)
            features = features_mean

        # print('features', features.shape, features.unique())
        # concat normalized depth value
        if concat_imz:
            im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
            im_z_mean = im_z[im_z > 0].mean()
            im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
            im_z_norm = (im_z - im_z_mean) / im_z_std
            im_z_norm[im_z <= 0] = 0
            # import ipdb; ipdb.set_trace()
            features = torch.cat([features, im_z_norm], dim=1)
        if torch.isnan(features).any().item():
            import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        feature_volume_all[batch_ind] = features

    # print('count', count.shape, count.unique())
    # print('feature_volume_all', feature_volume_all.shape, feature_volume_all.unique())
    
    return feature_volume_all, count


def back_project_visible(origin, voxel_size, n_vox, KRcam, feats):
    h, w = feats.shape[-2:]
    device  = feats.device
    n_views, bs, _, _ = KRcam.shape
    coords = generate_grid(n_vox, 1) #interval =1 
    coords_batch = coords.view(-1, 3)
    count = []
    for batch in range(bs):
        origin_batch = origin[batch].unsqueeze(0)
        proj_batch = KRcam[:, batch]

        if not isinstance(voxel_size, list):
            grid_batch = coords_batch * voxel_size + origin_batch.float()
        else:
            grid_batch = coords_batch * torch.tensor(voxel_size)[None].to(device) + origin_batch.float()
        
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).to(device)], dim=1)#(nview, 4, nV)

        # Project grid, nV=24/48/96^3
        im_p = proj_batch @ rs_grid #(nview, 4, nV)
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]#im_z exits 0
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0) #inside image

        mask = mask.view(n_views, -1)
        count.append(mask.sum(dim=0).float())

    count = torch.stack(count, dim=0)
    return count
