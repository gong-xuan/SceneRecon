import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import random


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, tsdf_path, nviews, n_scales, subset=1, max_depth=3):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.tsdf_path = tsdf_path
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        
        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 1
        #for fast validation
        self.subset = subset if self.mode=="val" else 1 

        self.metas = self.build_list()


    def build_list(self):
        with open(os.path.join(self.tsdf_path, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        
        if self.subset<1:   
            random.shuffle(metas)
            n_sset = int(len(metas)*self.subset)
            metas = metas[:n_sset]
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters->meters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):#0+1
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.tsdf_path, self.tsdf_file), meta['scene'])

        # import ipdb; ipdb.set_trace()
        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }
        # print('Before T', items['imgs'][0].size, items['depth'][0].shape)
        if self.transforms is not None:
            items = self.transforms(items)
        # print('After T', items['imgs'].shape, items['depth'].shape)
        return items




class ScanNetSceneDataset(Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, scene, data_path, max_depth, id_list=None):
        """
        Args:
        """
        n_imgs = len(os.listdir(f'{data_path}/{scene}/color'))
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", str(id) + ".txt"), delimiter=' ')
        # assert cam_pose[0,0] != np.inf and cam_pose[0,0] != -np.inf and cam_pose[0,0] != np.nan
        
        depth_name = os.path.join(self.data_path, self.scene, "depth", str(id) + ".png")
        depth_im = cv2.imread(depth_name, -1).astype(np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        # print('===================', depth_im.max(), depth_im.min())
        depth_im[depth_im > self.max_depth] = 0

        # Read RGB image
        color_name = os.path.join(self.data_path, self.scene, "color", str(id) + ".jpg")
        
        return cam_pose, depth_im, color_name