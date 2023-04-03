import sys
# sys.path.append('.')
sys.path.append('../..')
from datasets.sevenscenes import SevenSceneDataset
from torch.utils.data import DataLoader

import dataconfig
import torch
import os
import cv2
import numpy as np
VIS_PATH = './vis_coords'

def build_xyz_axis(volume_size=2500, device='cpu', homogenuous=False):
    x_axis_world = torch.zeros((volume_size, 3))
    y_axis_world = torch.zeros((volume_size, 3))
    z_axis_world = torch.zeros((volume_size, 3))
    for vn in range(volume_size):
        x_axis_world[vn, 0] = vn
        y_axis_world[vn, 1] = vn
        z_axis_world[vn, 2] = vn
    xyz_axis_world = torch.cat([x_axis_world[None], y_axis_world[None], z_axis_world[None]], dim=0)
    if homogenuous:
        add_ones = torch.ones((3, volume_size, 1))
        xyz_axis_world = torch.cat([xyz_axis_world, add_ones], dim=-1)#(3, npoint, 4)
    return xyz_axis_world.to(device)

def draw_axis_onimg(image, xyz_axis, savepath=None):
    """
    xyz_axis: (3, n_points, 2)
    """
    point_size=1
    point_color = [(0,0,255), (0,255,0), (255,0,0)]#BGR
    thickness = 4
    volume_size = xyz_axis.shape[1]
    imgh, imgw = image.shape[:2]
    for axisn in range(3):
        for vn in range(volume_size):
            u, v = xyz_axis[axisn, vn, :2]
            u, v = int(u), int(v)
            if u>=0 and u<imgw and v>=0 and v<imgh:
                print('draw points')
                image = cv2.circle(image, (u,v), point_size, point_color[axisn], thickness)
    if savepath:
        cv2.imwrite(savepath, image)
    else:
        return image

def draw_points_onimg(image, pts_list, savepath=None, point_color =(0,0,255)):
    """
    pts_list: 2D list: (npoints,2)
    """
    point_size=1
    #R
    thickness = 10
    imgh, imgw = image.shape[:2]
    for np in range(len(pts_list)):
        u, v = pts_list[np]
        u, v = int(u), int(v)
        print(u,v)
        if u>=0 and u<imgw and v>=0 and v<imgh:
            print('draw points')
            image = cv2.circle(image, (u,v), point_size, point_color, thickness)
    if savepath:
        cv2.imwrite(savepath, image)
    else:
        return image

def perspective_project_torch(points, rotation, translation, cam_K=None, returnZ=False, trans_first=True):
    """
    This function computes the perspective projection of a set of points in torch.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        cam_K (bs, 3, 3): Camera intrinsics matrix
    """
    batch_size = points.shape[0]
    # Transform points
    if trans_first:
        if translation is not None:
            points = points + translation.unsqueeze(1)
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    
    if not trans_first:
        if translation is not None:
            points = points + translation.unsqueeze(1)
    
    if cam_K is None:
        return points
    else:
        # Apply perspective distortion
        projected_points = points / (points[:, :, -1].unsqueeze(-1)+1e-8)
        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', cam_K, projected_points)
    if returnZ:
        return torch.cat([projected_points[:, :, :-1], points[:, :, -1].unsqueeze(-1)], dim=2)
    else:
        return projected_points[:, :, :-1]

def proj_xyz_img(cam_pose, intrinsics):
    # intrinsics = intrinsics.astype(np.float32)
    if True:
        xyz1_world = build_xyz_axis(homogenuous=True).permute(2,0,1).reshape(4, -1)/100. #(4, 3, np)->(4, 3*np)
        xyz1_world[-1] = 1.
        proj_mat = intrinsics[:3,:3] @ np.linalg.inv(cam_pose)[:3,:4]
        xyz1_img = proj_mat @ xyz1_world.numpy()
        xy_img = xyz1_img[:2]/xyz1_img[2][None]
        xy_img = xy_img.reshape(2, 3, -1)
        xy_img = np.transpose(xy_img, (1,2,0))
    else:
        xyz_axis_world = build_xyz_axis()/100.
        cam_K = intrinsics[:3, :3]
        cam_R = cam_pose[:3, :3].T
        cam_T = -cam_pose[:3, 3]
        xy_img = perspective_project_torch(xyz_axis_world, 
                        torch.tensor(cam_R).expand(3, -1, -1),
                        torch.tensor(cam_T).expand(3,-1),
                        cam_K = torch.tensor(cam_K).expand(3, -1, -1),
                        trans_first = True,
                        returnZ = False)#(3,Np, 2)
        xy_img = xy_img.numpy()
    return xy_img

def proj_xyz_world(xyz_cam,cam_pose, intrinsics):
    """
    xyz_cam: (np,4)
    """

    proj_mat = cam_pose @ np.linalg.inv(intrinsics) 

    xyz_world = proj_mat @ xyz_cam.T

    return xyz_world


def visualize_scannet(scene, img_id, savef, data_path=dataconfig.SCANNET_TEST_PATH):
    cam_pose = np.loadtxt(os.path.join(data_path, scene, 'frames', "pose", str(img_id) + ".txt"), delimiter=' ')#cam2world
    image = cv2.imread(os.path.join(data_path, scene, 'frames', "color", str(img_id) + ".jpg"))
    intrinsics = np.loadtxt(os.path.join(data_path, scene, 'frames', 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')
    xy_img = proj_xyz_img(cam_pose, intrinsics)
    draw_axis_onimg(image, xy_img, savef)

def visualize_7scenes(scene, img_id, savef, data_path=dataconfig.SEVEN_SCENES_TEST_PATH, corr_pose=None, vistype='coords'):
    imgf = f'{data_path}/{scene}/frame-000{img_id}.color.png'
    posef = imgf.replace('color.png', 'pose.txt')
    depthf = imgf.replace('color', 'depth')
    image = cv2.imread(imgf)
    cam_pose = np.loadtxt(posef).astype('float32')
    if corr_pose is not None:
        cam_pose = corr_pose @ cam_pose
    intrinsics = np.array([[585, 0 , 320, 0],
                            [0, 585, 240, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
    if vistype=='coords':
        xy_img = proj_xyz_img(cam_pose, intrinsics)
        draw_axis_onimg(image, xy_img, savef)
    elif vistype=='trans':
        uv_pts = [[100, 400], [200, 400], [150, 300], [200, 300]]
        draw_points_onimg(image, uv_pts, savef, point_color=(0,0,255))
        uv_pts = [[480, 400], [400, 200], [350, 300], [400, 300]]
        draw_points_onimg(image, uv_pts, savef, point_color=(0,255,0))
        import ipdb; ipdb.set_trace()

        # cv2.imwrite(savef, image)
        depth = cv2.imread(depthf,-1).astype(np.float32)/1000.
        xyz_cam = []
        for pt in uv_pts:
            d = depth[pt[1], pt[0]]
            xyz = [d*pt[0], d*pt[1], d ,1]
            xyz_cam.append(xyz)
        xyz_cam = np.array(xyz_cam)
        xyz_world = proj_xyz_world(xyz_cam, cam_pose, intrinsics)
        print(xyz_world)
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # data = 'scannet'
    data='7scenes'
    if data=='scannet':
        for imgid in range(1, 1000, 50):
            visualize_scannet('scene0806_00', imgid, f'./vis_coords/scannet{imgid}.png')
    else:
        scene = dataconfig.SEVEN_SCENES_TEST[0]
        #pip install SimpleITK
        import SimpleITK as sitk
        tsdf_path = dataconfig.SEVEN_SCENES_TSDF_PATH
        tsdf = sitk.GetArrayFromImage()
        #read raw
        # import ipdb; ipdb.set_trace()
        for scene in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
            data = np.fromfile(f'{tsdf_path}/{scene}.raw', dtype=np.int16)
            print(data.shape)
            data = sitk.ReadImage(f'{tsdf_path}/chess.mhd', sitk.sitkFloat32)
            print(data.GetOrigin(), data.GetSpacing())
        print(512*512*512)
        import ipdb; ipdb.set_trace()

        # print(np.sum(scene_image_array==np.nan))

        # corr_pose = np.array([[1, 0 , 0, 0],
        #                     [0, 0, 1, 0],
        #                     [0, -1, 0, 0],
        #                     [0, 0, 0, 1]], dtype=np.float32)
        #####################################################
        
        # MAX_DEPTH = 4
        # from datasets.transforms import ResizeImage, ToTensor, RandomTransformSpace, IntrinsicsPoseToProjection, Compose
        # transforms = Compose([ResizeImage((640, 480)), ##（W,H): same for sevenscenes
        #                         ToTensor(),
        #                         RandomTransformSpace( [96, 96, 96], 0.04, 
        #                         False, False, 0, 0, max_epoch=1, max_depth=MAX_DEPTH),
        #                     IntrinsicsPoseToProjection(9, 4),
        #                     ])
        
        # dataset = SevenSceneDataset('/data/7scenes', "test", transforms, 9, 2, max_depth=MAX_DEPTH, corr_pose=None)
        # dataLoader = DataLoader(dataset, 
        #                         1, 
        #                         shuffle=False, 
        #                         num_workers=1,
        #                         drop_last=False)
        # # depth_max = -np.inf
        # for batch_idx, sample in enumerate(dataLoader):
        #     # depth = sample['depth']
        #     # depth_max = max (depth.numpy().max(), depth_max)
        #     # print(depth_max)
        #     if not (sample['tsdf_list'][0].unique()<1).any():
        #         import ipdb; ipdb.set_trace()
        
        # for imgid in range(270, 320):
        #     # vistype = 'coords'
        #     vistype = 'trans'
        #     visualize_7scenes(scene, imgid, f'./vis_{vistype}/7scenes{imgid}-v2.png', corr_pose = corr_pose, vistype=vistype)


    