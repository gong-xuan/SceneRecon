import torch


def custom_meshgrid(*args):
    from packaging import version as pver
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def proj_rays_distort(pt2d, K, distCoef, c2w):
    # pt2d: tensor [2, N]
    # distCoef: tensor [5]
    # c2w: tenosr [4, 4]   
    N_points = pt2d.shape[1]
    K_inv = torch.linalg.inv(K.to(pt2d))
    unproj_pt = K_inv @ torch.cat([pt2d, torch.ones([1,N_points]).to(pt2d)], dim=0) #[3, N] third channel is one
    #undistort
    N_coef = distCoef.shape[0]
    k = torch.zeros([12])
    k[:N_coef] = distCoef[:N_coef]
    k.to(pt2d)
    x0, y0 = unproj_pt[0,:], unproj_pt[1,:]
    x, y = x0.clone(), y0.clone()
    for _ in range(N_coef):
        r2 = x*x + y*y
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
        deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist
    unproj_pt = torch.stack([x,y,torch.ones_like(x)]) # depth=1 
    # cam2world
    rays_dir = c2w[:3, :3] @ unproj_pt
    rays_ori = c2w[:3, 3].view([3,1]).expand(rays_dir.shape)
    rays_dir = rays_dir / torch.norm(rays_dir, dim=0, keepdim=True)
    return unproj_pt, rays_ori, rays_dir


def get_rays(c2w, intrinsics, H, W, distCoef=None, corner_only=False):
    """
    R,T: c2w
    T 
    """
    fx, fy, cx, cy = intrinsics
    if not corner_only:
        i, j = custom_meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
        i = i.t().reshape([H*W]) + 0.5 #(W,H)
        j = j.t().reshape([H*W]) + 0.5
    else:
        i = torch.tensor([0, 0, W-1, W-1]).float() + 0.5
        j = torch.tensor([0, H-1, 0, H-1]).float() + 0.5

    if distCoef is None:
        R = c2w[:3,:3]
        T = c2w[:3,3] 
        zs = torch.ones_like(i)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        directions = torch.stack((xs, ys, zs), dim=-1)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        rays_d = directions @ R.transpose(-1, -2) # (N, 3), directions are in cam
        rays_o = torch.as_tensor(T) # [3]
        rays_o = rays_o[None, :].expand_as(rays_d) # [N, 3] 
        # import ipdb; ipdb.set_trace()
    else:
        K = torch.tensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
        unproj_pt, rays_o, rays_d = proj_rays_distort(torch.stack([i,j]), K, torch.tensor(distCoef), c2w)
        rays_o = rays_o.permute(1,0)
        rays_d = rays_d.permute(1,0)
    return rays_o, rays_d