from datasets.colon3dv import Colon3DVDataset
import numpy as np
import open3d as o3d
import torch
from ops.tsdf_fusion import pcwrite, meshwrite
import cv2

D = np.asarray([[-0.18867185058223412],[-0.003927337093919806],[0.030524814153620117],[-0.012756926010904904]]).reshape([-1])

def rgbd2volume(dataloader, save_pcd='', save_mesh='', voxel_size=1, distort=1):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(voxel_size) ,
                sdf_trunc=3 * float(voxel_size) ,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for batch in dataloader:
        color_im = batch["image"][0].numpy()
        
        height, width = color_im.shape[:2]
        depth = batch["depth"][0].numpy()
        cam_pose_one = batch["c2w"][0].numpy()
        fx, fy = dataloader.dataset.focal
        cx, cy = dataloader.dataset.principal

        if distort:
            K = np.asarray([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
            color_im = cv2.fisheye.undistortImage(color_im, K, D=D, Knew=K)
        #

        # color_im = np.repeat(depth_one[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_im), 
                                                                o3d.geometry.Image(depth), 
                                                                depth_scale=1.0,
                                                                depth_trunc=120.,
                                                                convert_rgb_to_intensity=False)

        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(width=width, height=height, 
                                            fx=fx, fy=fy,
                                            cx=cx, cy=cy), 
            cam_pose_one)
    

    if save_mesh:
        mesh = volume.extract_triangle_mesh()
        mesh = mesh.compute_triangle_normals()

        meshwrite(f'{save_mesh}/mesh{voxel_size})_d{distort}.ply', np.asarray(mesh.vertices), 
            np.asarray(mesh.triangles), np.asarray(mesh.triangle_normals), 255*np.asarray(mesh.vertex_colors))
    if save_pcd:
        point_cloud = volume.extract_point_cloud()
        pcd = np.concatenate([np.asarray(point_cloud.points), np.asarray(point_cloud.colors)],1)
        pcwrite(f'{save_pcd}/pcd.ply', pcd)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    dataset = Colon3DVDataset()
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

    rgbd2volume(dataloader, voxel_size=1, save_pcd="", save_mesh="./output")