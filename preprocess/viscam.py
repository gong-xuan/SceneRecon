import numpy as np
import open3d as o3d
import sys
sys.path.append('../..')
import dataconfig
import os, cv2

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center', device='cuda'):
    import torch
    c2w = torch.tensor(c2w).to(device)
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device),
        torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, inverse_y, flip_x, flip_y, ndc=False, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    # if ndc:
    #     rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


def process(pose_list, HW_list, intrinsics_list, savef, inverse_y, far = 3., near = 0, ndc = False):
    cam_lst = []
    xyz_min = np.array([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min

    for c2w, (H, W), K in zip(pose_list, HW_list, intrinsics_list):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H, W, K, c2w, inverse_y=inverse_y, flip_x=False, flip_y=False, ndc=ndc)
        cam_o = rays_o[0,0].cpu().numpy()
        cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
        cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        #min, max
        rays_o = rays_o.cpu().numpy()
        rays_d = rays_d.cpu().numpy()
        viewdirs = viewdirs.cpu().numpy()
        if ndc:
            pts_nf = np.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = np.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        
        xyz_min = np.minimum(xyz_min, np.min(pts_nf,(0,1,2)))
        xyz_max = np.maximum(xyz_max, np.max(pts_nf,(0,1,2)))
    np.savez_compressed(savef,
        xyz_min=xyz_min, xyz_max=xyz_max, cam_lst=np.array(cam_lst))
    # import ipdb; ipdb.set_trace()

def visualize(path):
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('path')
    # args = parser.parse_args()

    data = np.load(path)
    xyz_min = data['xyz_min']
    xyz_max = data['xyz_max']
    cam_lst = data['cam_lst']

    # Outer aabb
    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])
    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

    # Cameras
    cam_frustrm_lst = []
    for cam in cam_lst:
        cam_frustrm = o3d.geometry.LineSet()
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
        if len(cam) == 5:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(8)])
            cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])
        elif len(cam) == 8:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0,0,0] for i in range(12)])
            cam_frustrm.lines = o3d.utility.Vector2iVector([
                [0,1],[1,3],[3,2],[2,0],
                [4,5],[5,7],[7,6],[6,4],
                [0,4],[1,5],[3,7],[2,6],
            ])
        else:
            raise NotImplementedError
        cam_frustrm_lst.append(cam_frustrm)

    # Show
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0,0,0])),
        out_bbox, *cam_frustrm_lst])


def presave_scannet(scene, save_path, downsample=10):
    from corr_7scenes import draw_axis_onimg, proj_xyz_img
    scene_path = f'{dataconfig.SCANNET_TEST_PATH}/{scene}'
    image_path = f'{scene_path}/frames/color'
    pose_path = f'{scene_path}/frames/pose'
    intrinsics_file = f'{scene_path}/frames/intrinsic/intrinsic_color.txt'
    intrinsics = np.loadtxt(intrinsics_file, delimiter=' ')
    #depth_path
    pose_list = []
    HW_list = []
    intrinsics_list = []
    for imgf in os.listdir(image_path)[::downsample]:
        image = cv2.imread(f'{image_path}/{imgf}')
        HW = image.shape[:2]
        posef = imgf.replace('jpg', 'txt')
        cam_pose = np.loadtxt(f'{pose_path}/{posef}', delimiter=' ')#cam2world
        #
        pose_list.append(cam_pose)
        HW_list.append(HW)
        intrinsics_list.append(intrinsics)
        #save image
        xy_img = proj_xyz_img(cam_pose, intrinsics)
        draw_axis_onimg(image, xy_img, f'{save_path}/{imgf}')
    process(pose_list, HW_list, intrinsics_list, f'{save_path}/scannet_{scene}_d{downsample}.npz', inverse_y=True, far=3.)
    
def presave_7scenes(scene, save_path, downsample=10):
    from corr_7scenes import draw_axis_onimg, proj_xyz_img

    scene_path = f'{dataconfig.SEVEN_SCENES_TEST_PATH}/{scene}'
    intrinsics = np.array([[585, 0 , 320],
                            [0, 585, 240],
                            [0, 0, 1]], dtype=np.float32)
    corr_pose = np.array([[1, 0 , 0, 4.],
                        [0, 0, 1, 2.],
                        [0, -1, 0, 2.],
                        [0, 0, 0, 1]], dtype=np.float32)
    #
    pose_list = []
    HW_list = []
    intrinsics_list = []
    img_list = [file for file in os.listdir(scene_path) if file.endswith('color.png') ] 
    img_list = sorted(img_list)
    for imgf in img_list[::downsample]:
        image = cv2.imread(f'{scene_path}/{imgf}')
        HW = image.shape[:2]
        posef = imgf.replace('color.png', 'pose.txt')
        cam_pose = np.loadtxt(f'{scene_path}/{posef}').astype('float32')#delimiter=' ' if converting string to float
        cam_pose = corr_pose @ cam_pose
        pose_list.append(cam_pose)
        HW_list.append(HW)
        intrinsics_list.append(intrinsics)
        #save image
        xy_img = proj_xyz_img(cam_pose, intrinsics)
        draw_axis_onimg(image, xy_img, f'{save_path}/{imgf}')
    scene = scene.replace('/', '-')
    process(pose_list, HW_list, intrinsics_list, f'{save_path}/{scene}_d{downsample}.npz', inverse_y=True, far=3.)
    

if __name__ == '__main__':
    # data = 'scannet'
    data='7scenes'
    if data=='scannet':
        scene = 'scene0806_00'
        # presave_scannet(scene,  f'./vis_cam')
        visualize(f'./vis_cam/scannet_{scene}_d10.npz')
    elif data=='7scenes':
        scene = dataconfig.SEVEN_SCENES_TEST[0]
        # presave_7scenes(scene, f'./vis_cam7s' )
        scene = scene.replace('/', '-')
        visualize(f'./vis_cam7s/{scene}_d10.npz')
