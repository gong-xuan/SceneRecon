import os, cv2, torch, trimesh
import numpy as np
from skimage import measure
import torchvision.utils as vutils
from loguru import logger
import json

from ops.pyrender import Visualizer
from utils.utils import tensor2numpy, tensor2float

def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


def print_metrics(fname):
    key_names = ['AbsRel', 'AbsDiff', 'SqRel', 'RMSE', 'LogRMSE', 'r1', 'r2', 'r3', 'complete', 'dist1', 'dist2',
                 'prec', 'recal', 'fscore']

    metrics = json.load(open(fname, 'r'))
    metrics = sorted([(scene, metric) for scene, metric in metrics.items()], key=lambda x: x[0])
    scenes = [m[0] for m in metrics]
    metrics = [m[1] for m in metrics]

    keys = metrics[0].keys()
    metrics1 = {m: [] for m in keys}
    for m in metrics:
        for k in keys:
            metrics1[k].append(m[k])

    metrics2 = {}
    for k in key_names:
        if k in metrics1:
            v = np.nanmean(np.array(metrics1[k]))
        else:
            v = np.nan
        print('%10s %0.3f' % (k, v))
        metrics2[k] = v
    return metrics2

class SaveScene(object):
    def __init__(self, cfg, save_path):
        self.cfg = cfg
        self.scene_name = None
        self.global_origin = None
        self.tsdf_volume = []  # not used during inference.
        self.weight_volume = []

        self.coords = None
        self.keyframe_id = None

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path

        if cfg.VIS_INCREMENTAL:
            self.vis = Visualizer()

    def close(self):
        self.vis.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.keyframe_id = 0
        self.tsdf_volume = []
        self.weight_volume = []

        # self.coords = coordinates(np.array([416, 416, 128])).float()

        # for scale in range(self.cfg.MODEL.N_LAYER):
        #     s = 2 ** (self.cfg.MODEL.N_LAYER - scale - 1)
        #     dim = tuple(np.array([416, 416, 128]) // s)
        #     self.tsdf_volume.append(torch.ones(dim).cuda())
        #     self.weight_volume.append(torch.zeros(dim).cuda())

    @staticmethod
    def tsdf2mesh(voxel_size, origin, tsdf_vol):
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
        return mesh

    def vis_incremental(self, batch_idx, imgs, outputs):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # vis
            key_frames = []
            for img in imgs[::3]:
                img = img.permute(1, 2, 0)
                img = img[:, :, [2, 1, 0]]
                img = img.data.cpu().numpy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                key_frames.append(img)
            key_frames = np.concatenate(key_frames, axis=0)
            cv2.imshow('Selected Keyframes', key_frames / 255)
            cv2.waitKey(1)
            # vis mesh
            self.vis.vis_mesh(mesh)

    def save_incremental(self, save_path, batch_idx, imgs, outputs):
        # save_path = os.path.join(self.log_dir + '_' + 'incremental_' + str(epoch_idx), self.scene_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save
            mesh.export(os.path.join(save_path, 'mesh_{}.ply'.format(self.keyframe_id)))

    def save_scene_eval(self, scene_name, outputs, batch_idx=0):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(scene_name))
        else:
            # Marching cubes
            # import ipdb; ipdb.set_trace()
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)#marching cube and translate/scale
            # save tsdf volume for atlas evaluation
            data = {'origin': origin,
                    'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                    'tsdf': tsdf_volume}
            
            np.savez_compressed(
                os.path.join(self.save_path, '{}.npz'.format(scene_name)),
                **data)
            mesh.export(os.path.join(self.save_path, '{}.ply'.format(scene_name)))
        # self.save_path = save_path

    def __call__(self, outputs):
        # no scene saved, skip
        if "scene_name" not in outputs.keys():
            return
        batch_size = len(outputs['scene_name'])
        for i in range(batch_size):
            scene = outputs['scene_name'][i]
            scene_name = scene.replace('/', '-')
            self.save_scene_eval(scene_name, outputs, i)