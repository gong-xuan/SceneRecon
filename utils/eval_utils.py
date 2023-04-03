from loguru import logger
import os
import numpy as np
import json
import torch
import trimesh
import open3d as o3d
from ops.p3drender import Renderer_Pytorch3D
from datasets.scannet import ScanNetSceneDataset
from datasets.sevenscenes import  SevenScenesSceneDataset
from utils.metrics_utils import eval_depth, eval_depth_batch, eval_mesh
import time
from utils.utils import DictAverageMeter


class EvalMetrics():
    def __init__(self, cfg, device, mean='opt', nanmean=False, eval3D=False, loader_num_workers=8, render_bs=64):
        """
        nanmean= False is original version
        """
        self.cfg = cfg
        self.data_path = cfg.TEST.PATH
        self.max_depth = 10
        self.loader_num_workers = loader_num_workers
        #
        self.eval3D = eval3D
        self.width = 640
        self.height = 480
        #evaluation
        self.device = device
        self.render_bs = render_bs
        # self.nanmean = nanmean #TODO
        # self.mean = mean


    def __call__(self, log_dir, scene_list):
        self.log_dir = log_dir
        # metric = DictAverageMeter()
        metrics_all = {}    

        for scene_idx, scene in enumerate(scene_list):
            start_time = time.time()
            metric_scene = self.eval_per_scene(scene)
            print(f"{scene},{scene_idx+1}/{len(scene_list)}:", time.time()-start_time)
            # metric.update(metric_scene)
            metrics_all[scene] = metric_scene

        # for key, value in metric.mean().items():
        #     print('%10s %0.3f' % (key, value))
        
        # metrics_all['mean'] = metric.mean()
        return metrics_all
        

    def init_per_scene(self, scene, voxel_size=4):
        if self.eval3D:
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(voxel_size) / 100,
                sdf_trunc=3 * float(voxel_size) / 100,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            volume = None
        
        #Dataloader
        dataset = ScanNetSceneDataset(scene, self.data_path, self.max_depth)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=self.render_bs, 
                                                # collate_fn=collate_fn,#batch_sampler=None, 
                                                drop_last = False,
                                                num_workers=self.loader_num_workers)
        
        #Render
        cam_intr = np.loadtxt(
                        f'{self.data_path}/{scene}/intrinsic/intrinsic_depth.txt', 
                        delimiter=' ')[:3, :3]
        mesh_file = os.path.join(self.log_dir, '%s.ply' % scene.replace('/', '-'))
        mesh = trimesh.load(mesh_file, process=False)#vertices and faces
        renderer = Renderer_Pytorch3D(self.render_bs, 
                                        device=self.device, 
                                        height=self.height, 
                                        width=self.width)

        renderer.set_intrinsics_mesh(cam_intr, mesh)

        return volume, dataloader, cam_intr, renderer


    def filter_nanpose(self, cam_pose, depth_trgt):
        if torch.isinf(cam_pose[:,0,0]).any().item() or torch.isnan(cam_pose[:,0,0]).any().item():
            isinf = torch.isinf(cam_pose[:,0,0])
            isnan = torch.isnan(cam_pose[:,0,0])
            pop = isinf + isnan #or
            #filter out
            if pop.all().item():
                return None, None
            else:
                return cam_pose[~pop], depth_trgt[~pop]
        else:
            return cam_pose, depth_trgt


    def eval_per_scene(self, scene):
        volume, dataloader, cam_intr, renderer = self.init_per_scene(scene)
        metrics2DMeter = DictAverageMeter()
        for (cam_pose, depth_trgt, _) in dataloader:
            """
            cam_pose: tensor (bs, 4,4)
            depth_trgt: tensor (bs, h, w)
            """
            cam_pose, depth_trgt = self.filter_nanpose(cam_pose, depth_trgt)
            if cam_pose is None and depth_trgt is None:
                continue
            depth_pred = renderer(cam_pose).detach().cpu().numpy()
            batchsize = depth_pred.shape[0]
            depth_trgt = depth_trgt.numpy()
            cam_pose = cam_pose.numpy()
            
            # metrics2D.update(eval_depth_batch(depth_pred, depth_trgt), new_count=batchsize)
            for n in range(batchsize):
                temp = eval_depth(depth_pred[n], depth_trgt[n])
                metrics2DMeter.update(temp)

            if self.eval3D:
                for n in range(batchsize):
                    depth_one = depth_pred[n]
                    cam_pose_one = cam_pose[n]

                    color_im = np.repeat(depth_one[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_im), 
                                                                            o3d.geometry.Image(depth_one), 
                                                                            depth_scale=1.0,
                                                                            depth_trunc=5.0,
                                                                            convert_rgb_to_intensity=False)

                    volume.integrate(
                        rgbd,
                        o3d.camera.PinholeCameraIntrinsic(width=self.width, height=self.height, 
                                                        fx=cam_intr[0, 0], fy=cam_intr[1, 1],
                                                        cx=cam_intr[0, 2], cy=cam_intr[1, 2]), 
                        np.linalg.inv(cam_pose_one))
        
        # import ipdb; ipdb.set_trace()
        # clear
        # del self.renderer.rasterizer
        # del self.renderer.faces_list
        # del self.renderer.vertices_ts

        # all fragments of one scene completed
        if self.eval3D:
            verts_pred = volume.extract_triangle_mesh().vertices #np.asarray(volume.extract_triangle_mesh().vertices)
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(verts_pred)

            pcd_trgt = o3d.io.read_point_cloud(f'{self.cfg.TEST.PATH}/{scene}/{scene}_vh_clean_2.ply')
            metrics3D = eval_mesh(pcd_pred, pcd_trgt, pcd_loaded=True)
            
            # import ipdb; ipdb.set_trace()
            # pcd_pred2 = o3d.geometry.PointCloud()
            # pcd_pred2.points = o3d.utility.Vector3dVector(mesh.vertices)
            # metrics3D2 = eval_mesh(pcd_pred2, pcd_trgt, pcd_loaded=True)

        metrics2D = metrics2DMeter.mean()
        
        return {**metrics2D, **metrics3D}
        
         
        
        
        