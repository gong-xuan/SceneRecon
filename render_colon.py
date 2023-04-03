from datasets.colon3dv import Colon3DVDataset
import numpy as np
import os
import torch
from ops.tsdf_fusion import pcwrite, meshwrite
import cv2
import trimesh
from ops.p3drender import Renderer_Pytorch3D
import matplotlib.pyplot as plt
import imageio
from utils.ray_utils import get_rays
from utils.distort_utils import distort_cam_pts_torch
from utils.rot_utils import rot_x, rot_y, rot_z

def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


def render_w_gt(renderer, dataloader, render_path):
    rgbs = []
    for id, (cam_pose, depth_im, img_file) in enumerate(dataloader):
        color_im = cv2.imread(img_file).astype(np.float32)
        # color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

        depth_r, color_r = renderer(torch.tensor(cam_pose)[None], render_rgb=True)
        depth_r = depth_r[0].cpu().numpy()
        color_r = color_r[0].cpu().numpy()[...,::-1]*255 #rgb2bgr
        
        color_concat = np.concatenate([color_r, color_im], axis=1)
        rgbs.append(color_concat)
        # cv2.imwrite(f'{render_path}/{id}_color_pred.jpg', color_r)
        # plt.imsave(f'{render_path}/{id}_d_pred.jpg', depth_r) # img: [n,n]
        # plt.imsave(f'{render_path}/{id}_d_gt.jpg', depth_im) 

    rgbs = np.array(rgbs)[...,::-1]
    imageio.mimwrite(f'{render_path}/color.mp4', rgbs.astype(np.uint8), fps=10, quality=8)


def build_cam_lines(rays_o, rays_d, far, pts_interval=0.5):
    assert rays_o.shape[0] == 4 and rays_d.shape[0] == 4
    pts_list = []
    # 4 lines from rays_0
    for nc in range(4):
        rayo, rayd = rays_o[nc], rays_d[nc]
        for d in torch.arange(0, far, pts_interval):
            pts_list.append(rayo + d*rayd)
    #bbox at rays end
    
    rays_e = rays_o + far*rays_d #(4,3)
    ltop, lbot, rtop, rbot = rays_e[0], rays_e[1], rays_e[2], rays_e[3]
    for line in [[ltop, lbot], [ltop, rtop], [rtop, rbot], [rbot,lbot]]:
        pt1, pt2 = line
        dir = (pt1-pt2)/torch.norm(pt1-pt2)
        dis = torch.norm(pt1-pt2)
        for d in torch.arange(0, dis, pts_interval):
            pts_list.append(pt2 + d*dir)
    
    return torch.stack(pts_list)

def draw_cam_onimg(image, rays_o, rays_d, far, intrinsics, distCoef=None, pts_interval=0.5, color=(0,0,255), point_size=1, thickness = 1, savepath=None):
    """
    color: BGR
    """
    pts = build_cam_lines(rays_o, rays_d, far, pts_interval=pts_interval)
    if distCoef is not None:
        pts = distort_cam_pts_torch(pts, distCoef)
    
    imgh, imgw = image.shape[:2]
    fx, fy, tx, ty = intrinsics

    for np in range(pts.shape[0]):
        x, y ,z = pts[np]
        u, v = fx*x/z + tx, fy*y/z + ty
        u, v = int(u), int(v)
        if u>=0 and u<imgw and v>=0 and v<imgh:
            image = cv2.circle(image, (u,v), point_size, color, thickness)
    if savepath:
        cv2.imwrite(savepath, image)
    else:
        return image

def render_w_topview(renderer, dataloader, render_path, intrinsics, view_dist=False, cam_dist=True):
    v2w = torch.tensor([[1, 0, 0, 50],
                        [0, 1, 0, 50],
                        [0, 0, 1, -120],
                        [0, 0, 0, 1]]).float()
    depth_r, color_r = renderer(torch.tensor(v2w)[None], render_rgb=True, dist=view_dist)
    image = 255*color_r[0].cpu().numpy()[...,::-1]
    # cv2.imwrite(f'{render_path}/top.jpg', image)

    height = renderer.height
    width = renderer.width
    for id, (cam_pose, depth_im, img_file) in enumerate(dataloader):
        if not id%10==0:
            continue
        cam_pose = torch.tensor(cam_pose)
        c2v = torch.linalg.inv(v2w) @ cam_pose #c2v= w2v@c2w
        rays_o, rays_d = get_rays(c2v, intrinsics, height, width, distCoef=dataset.distcoef if cam_dist else None, 
            corner_only=True)# persample, v-coordinate
        draw_cam_onimg(image.copy(), rays_o, rays_d, 10, intrinsics, distCoef=dataset.distcoef if view_dist else None, 
            pts_interval=0.1, savepath=f'{render_path}/top_c{id}.jpg')
        
    import ipdb; ipdb.set_trace()

def render_w_sideview(renderer, dataloader, render_path, intrinsics, view_dist=False, cam_dist=False):
    # renderer.textures=None
    w2v = rot_x(np.pi*90/180.)
    w2v = w2v@torch.tensor([[1, 0, 0, -60],
                        [0, 1, 0, -200],
                        [0, 0, 1, 60],
                        [0, 0, 0, 1]]).float()
    depth_r, color_r, normal_r = renderer(torch.linalg.inv(w2v)[None], render_rgb=True, render_normal=True, dist=view_dist)
    image = (255*(normal_r[0]-normal_r[0].min())/(normal_r[0].max()-normal_r[0].min())).cpu().numpy()
    import ipdb; ipdb.set_trace()
    # image = 255*color_r[0].cpu().numpy()[...,::-1]
    cv2.imwrite(f'{render_path}/side_gt.jpg', image)

    height = renderer.height
    width = renderer.width
    for id, (cam_pose, depth_im, img_file) in enumerate(dataloader):
        if not id%10==0:
            continue
        cam_pose = torch.tensor(cam_pose)
        c2v = w2v @ cam_pose #c2v= w2v@c2w
        rays_o, rays_d = get_rays(c2v, intrinsics, height, width, distCoef=dataset.distcoef if cam_dist else None, 
            corner_only=True)# persample, v-coordinate
        draw_cam_onimg(image.copy(), rays_o, rays_d, 10, intrinsics, distCoef=dataset.distcoef if view_dist else None, 
            pts_interval=0.1, savepath=f'{render_path}/side_c{id}.jpg')
        
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    root_path = "/data_local/endo_datasets/colon3dv"
    root_path = "~/data/colon3dv"
    root_path = os.path.expanduser(root_path)

    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    dataset = Colon3DVDataset(root_path=f"{root_path}/reg_videos", video='c1v1')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=1)

    #
    mesh_file = f"{root_path}/all_tsdf/c1v1/mesh.ply"
    vertrgb_file = f"{root_path}/all_tsdf/c1v1/vert_rgb.npy"
    mesh = trimesh.load(mesh_file, process=False)#vertices and faces
    verts_rgb_colors = np.load(vertrgb_file).astype(float)/255.

    renderer = Renderer_Pytorch3D(1, 
                                device="cuda", 
                                height=1080, 
                                width=1350)

    fx, fy = dataset.focal
    tx, ty = dataset.principal
    cam_intr = np.array([[fx, 0, tx], [0, fy, ty], [0, 0, 1]])
    renderer.set_intrinsics_mesh(cam_intr, mesh, verts_rgb_colors=verts_rgb_colors, distCoef=dataset.distcoef)

    render_path = f"{root_path}/render/c1v1"
    os.makedirs(render_path, exist_ok=True)
    
    # render_w_gt(renderer, dataloader, render_path)
    # render_w_topview(renderer, dataloader, render_path, [fx,fy,tx,ty])
    render_w_sideview(renderer, dataloader, render_path, [fx,fy,tx,ty])
