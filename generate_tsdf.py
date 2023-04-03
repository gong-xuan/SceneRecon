import time
import pickle
import argparse
from tqdm import tqdm
import trimesh
import os
import numpy as np
from config import cfg
from ops.tsdf_fusion import TSDFVolume, get_view_frustum
from datasets.scannet import ScanNetSceneDataset
from datasets.colon3dv import Colon3DVDataset
# from datasets.sevenscenes import  SevenScenesSceneDataset
import cv2
import ray
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='neuralrecon',
                        type=str)
    parser.add_argument('--test', action='store_true',
                        help='prepare the test set')
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=1., type=float)

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=1, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8) 
    parser.add_argument('--gpu', type=str, default='0')
    
    return parser.parse_args()


args = parse_args()

@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def multiprocess_colon3dv(args, scene_videos, distort=True, use_gpu=False):
    process_colon3dv(args, scene_videos, distort=distort, use_gpu=use_gpu)

def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _

def save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, distCoef=None, use_gpu=True, save_mesh=False):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))

    n_imgs = len(depth_list.keys())
    if n_imgs > 500:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    for id in image_id:
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose, distCoef=distCoef)#project cornder to 3D world
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    
    # Integrate
    tsdf_vol_list = []
    # import ipdb; ipdb.set_trace()
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, use_gpu=use_gpu, margin=args.margin))

    # Loop through RGB-D images and fuse them together
    # t0_elapse = time.time()
    for id in depth_list.keys():
        # if id % 100 == 0:
        #     print("{}: Fusing frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        print(f"{scene_path}: Fusing frame {id}/{n_imgs}")
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]
        
        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1., distCoef=distCoef)

    # fps = n_imgs / (time.time() - t0_elapse)
    # print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
        # import ipdb; ipdb.set_trace()
        # print(np.unique(tsdf_vol))
        np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    if save_mesh:
        l = 0 #finest level
        #vol_bnds, voxel_size=args.voxel_size * 2 ** l
        verts, faces, norms, colors = tsdf_vol_list[l].get_mesh(scene_path)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
        mesh.export(f'{args.save_path}/{scene_path}/mesh.ply')
        np.save(f'{args.save_path}/{scene_path}/vert_rgb.npy', colors)
        print('=====Saved',args.save_path, scene_path)
        # point_cloud = tsdf_vol_list[l].get_point_cloud()
        # pcwrite(f'{args.save_path}/{scene_path}/pc_layer{l}.ply', point_cloud)

def save_fragment_pkl(args, scene, cam_intr, depth_list, cam_pose_list, imgfile_list):
    fragments = []
    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_imgfiles = []
    imgfiles = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            imgfiles.append(imgfile_list[id])
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                imgfiles.append(imgfile_list[id])
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    all_imgfiles.append(imgfiles)
                    ids = []
                    imgfiles = []
                    count = 0

    with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    for i, bnds in enumerate(all_bnds):
        if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'image_files':all_imgfiles[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })
    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)
    
    print(f'Completed: process scene {scene}, {count+len(all_bnds)*args.window_size} keyframes, {(len(all_bnds))} fragments')

def process_colon3dv(args, scene_videos, distort=True, use_gpu=False):
    for scene in tqdm(scene_videos):
        depth_all = {}
        cam_pose_all = {}
        color_all = {}
        imgfile_all = {}
            
        dataset = Colon3DVDataset(root_path="/data_local/endo_datasets/colon3dv/reg_videos", video=scene)
        fx, fy = dataset.focal
        tx, ty = dataset.principal
        cam_intr = np.array([[fx, 0, tx], [0, fy, ty], [0, 0, 1]])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=args.loader_num_workers)

        for id, (cam_pose, depth_im, img_file) in enumerate(dataloader):
            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            color_image = cv2.imread(img_file).astype(np.float32)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth_all.update({id: depth_im})
            cam_pose_all.update({id: cam_pose})
            imgfile_all.update({id:img_file})
            color_all.update({id: color_image}) #TODO

        distCoef = dataset.distcoef if distort else None
        save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, distCoef=distCoef, use_gpu=use_gpu, save_mesh=True)
        save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all, imgfile_all)

def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret

if __name__ == "__main__":
    cfg.merge_from_file(f'./config/{args.cfg}.yaml')
    
    args.save_path = os.path.join(cfg.TSDF_ROOT, "all_tsdf")
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    # files = ['c2v1', 'c2v2', 'c2v3', 'c3v2', 'c4v2']
    #, 'c4v3', 'd4v2', 's1v3', 's2v1', 's3v1', 's3v2', \
    #    't1v1', 't1v3', 't2v1', 't2v2', 't2v3', 't3v2', 't3v3', 't4v1', 't4v3']
    files = ['t1v1', 't1v3', 's1v3']
    files = ['c2v1', 'c2v2', 'c2v3']
    files = ['t2v1', 't2v2', 't2v3']
    files = ['s2v1', 's3v1', 's3v2']
    files = ['c4v2', 'd4v2']
    files = ['t4v1', 't4v3']
    all_proc = args.n_proc * args.n_gpu

    # process_colon3dv(args, ['c1v1', 'c1v3'], distort=True, use_gpu=False) #color, distortion only support CPU
    # process_colon3dv(args, ['c2v1', 'c2v2', 'c2v3', 'c3v2', 'c4v2', 'c4v3', 'd4v2', 's1v3', 's2v1', 's3v1', 's3v2', \
    #     't1v1', 't1v3', 't2v1', 't2v2', 't2v3', 't3v2', 't3v3', 't4v1', 't4v3'], 
    #     distort=True, use_gpu=False)

    if all_proc>1:
        ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu, ignore_reinit_error=True)
        files = split_list(files, all_proc)
        ray_worker_ids = []
        for w_idx in range(all_proc):
            ray_worker_ids.append(multiprocess_colon3dv.remote(args, files[w_idx], distort=True, use_gpu=False))

        results = ray.get(ray_worker_ids)
        ray.shutdown()
    else:
        process_colon3dv(args, files, distort=True, use_gpu=False)