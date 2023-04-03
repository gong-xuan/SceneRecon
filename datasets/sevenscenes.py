import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import random


_parse_fns = {
    'NDims': int,
    'DimSize': lambda x: tuple(int(k) for k in x.split()),
    'Offset': lambda x: tuple(int(k) for k in x.split()),
    'ElementSpacing': lambda x: tuple(float(k) for k in x.split()),
}

_dtypes = {
    'MET_SHORT': np.int16
}


class Tsdf(object):
    def __init__(self, offset, spacing, data, invalid_depth = 65535):
        self.offset = np.array(offset)
        self.spacing = np.array(spacing)
        # data[data == 0] = -16384
        self.data = data
        self.shape = data.shape
        self.invalid_depth = invalid_depth

    def get_signed_distance(self, xyz, out=None):
        data = self.data
        if out is None:
            out = np.empty(shape=xyz.shape[:-1], dtype=np.int32)

        ijk = self.xyz_to_ijk(xyz).astype(np.int32)
        valid = np.all([ijk >= 0, ijk < self.shape], axis=(0, -1))
        i, j, k = ijk[valid].T
        out[valid] = data[i, j, k]
        invalid = np.logical_not(valid)
        i, j, k = ijk[invalid].T
        out[invalid] = self.invalid_depth
        return out

    def xyz_to_ijk(self, xyz):
        xyz = xyz * 1000  # m to mm
        return (xyz + self.offset) / self.spacing

    def ijk_to_xyz(self, ijk):
        xyz = ijk * self.spacing - self.offset
        xyz /= 1000  # mm to m
        return xyz


class TsdfMeta(object):
    def __init__(self, n_dims, shape, offset, spacing, dtype, data_filename):
        self.n_dims = n_dims
        self.shape = shape
        self.offset = np.array(offset, dtype=np.float32)
        self.spacing = np.array(spacing, dtype=np.float32)
        self.dtype = dtype
        self.data_filename = data_filename

    @staticmethod
    def from_file(path):
        with open(path, 'r') as fp:
            data = dict()
            for line in fp.readlines():
                line = line.rstrip()
                if len(line) > 0:
                    key, value = line.split(' = ')
                    data[key] = value
        return TsdfMeta.from_raw(**data)

    @staticmethod
    def from_raw(
            NDims, DimSize, Offset, ElementSpacing, ElementType,
            ElementDataFile):
        return TsdfMeta(
            n_dims=int(NDims),
            shape=tuple(int(k) for k in DimSize.split()),
            offset=tuple(int(k) for k in Offset.split()),
            spacing=tuple(float(k) for k in ElementSpacing.split()),
            dtype=_dtypes[ElementType],
            data_filename=ElementDataFile)


def load_tsdf(scene_id, folder):
    meta = TsdfMeta.from_file(os.path.join(folder, '%s.mhd' % scene_id))
    data = np.fromfile(
        os.path.join(folder, meta.data_filename), dtype=meta.dtype)
    return Tsdf(meta.offset, meta.spacing, data.reshape(meta.shape))
    # return meta, data.reshape(meta.shape)


class SevenSceneDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales, max_depth=8, corr_pose=None, subset=1):
        super(SevenSceneDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["val","test"]#["train", "val", "test"]
        
        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 1
        #for fast validation
        self.subset = subset if self.mode=='val' else 1
        self.metas = self.build_list()
        self.max_depth = max_depth

        self.intrinsics = np.array([[585, 0 , 320],
                                    [0, 585, 240],
                                    [0, 0, 1]], dtype=np.float32)
        # self.corr_pose = np.array([[0, -1, 0, 0],
        #                             [1, 0, 0, 0],
        #                             [0, 0, 1, 0],
        #                             [0, 0, 0, 1]], dtype=np.float32)#Rz(90)
        # self.corr_pose = np.array([[0, 0 , -1, 0],
        #                             [0, 1, 0, 0],
        #                             [1, 0, 0, 0],
        #                             [0, 0, 0, 1]], dtype=np.float32)#Ry(-90)
        if corr_pose is None:
            self.corr_pose = np.array([[1, 0 , 0, 2],
                                    [0, 0, 1, 0.5],
                                    [0, -1, 0, 0.7],
                                    [0, 0, 0, 1]], dtype=np.float32)
            # self.corr_pose = np.array([[1, 0 , 0, 4.0],
            #                             [0, 0, 1, 2.],
            #                             [0, -1, 0, 2.],
            #                             [0, 0, 0, 1]], dtype=np.float32)#world_7s to world_scannet
        else:
            self.corr_pose = corr_pose
        # self.T0 = np.array([[ 1., 0., 0., 0.],
        #                     [ 0., 0., 1., 0.],
        #                     [ 0., -1., 0., 0.],
        #                     [ 0., 0., 0., 1.]], dtype=np.float32)#cam_openCV(scannet) to cam_openGL
        # self.T0 = self.T0.T

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        
        if self.subset<1:   
            random.shuffle(metas)
            n_sset = int(len(metas)*self.subset)
            metas = metas[:n_sset]
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath):
        extrinsics = np.loadtxt(filepath).astype('float32')
        extrinsics = self.corr_pose @ extrinsics #@self.T0
        return  extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters->meters
        # print('============', self.max_depth, depth_im.max(), depth_im.min())
        depth_im[depth_im > self.max_depth] = 0
        # print('============', depth_im.max())
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):#2+1
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

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])
        # for i, vid in enumerate(meta['image_ids']):
        for i, imgf in enumerate(meta['image_files']):
            # scene = meta['scene']
            # load images
            imgs.append(self.read_img(imgf))
            depthf = imgf.replace('color', 'depth')
            depth.append(self.read_depth(depthf))
            camf = imgf.replace('color.png', 'pose.txt')
            extrinsics = self.read_cam_file(camf)

            intrinsics_list.append(self.intrinsics)
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


class SevenScenesSceneDataset(Dataset):
    def __init__(self, scene_path, max_depth):
        """
        Args:
        """
        self.scene_path = scene_path
        self.imgs = [file for file in os.listdir(scene_path) if file.endswith('color.png') ]
        self.imgs.sort()
        self.max_depth = max_depth
        self.corr_pose = np.array([[1, 0 , 0, 2.0],
                                    [0, 0, 1, 0.5],
                                    [0, -1, 0, 0.7],
                                    [0, 0, 0, 1]], dtype=np.float32)
        self.corr_cam = np.array([[ 1., 0., 0., 0.],
                                [ 0., 0., 1., 0.],
                                [ 0., -1., 0., 0.],
                                [ 0., 0., 0., 1.]], dtype=np.float32)#cam_openCV(scannet) to cam_openGL
        self.corr_cam = self.corr_cam.T
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        img_file = f'{self.scene_path}/{self.imgs[id]}'
        depth_file = img_file.replace('color', 'depth')
        pose_file = img_file.replace('color', 'pose')[:-4] + '.txt'
        # 
        cam_pose = np.loadtxt(pose_file).astype('float32')#delimiter=' ' if converting string to float
        cam_pose = self.corr_pose @ cam_pose
        # print("CORR", self.corr_pose)

        # import ipdb; ipdb.set_trace() 
        # if np.any(cam_pose[0,0] == np.inf) or np.any(cam_pose[0,0] == -np.inf) or np.any(cam_pose[0,0] == np.nan):
        #     import ipdb; ipdb.set_trace() 
        # print("------", cam_pose.shape)
        # Read depth image and camera pose
        # print(depth_file)
        depth_im = cv2.imread(depth_file, -1).astype(np.float32) 
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        # import ipdb; ipdb.set_trace()
        # print(depth_im.max(), depth_im.min())
        depth_im[depth_im > self.max_depth] = 0
        # print(depth_im.max(), depth_im.min())
        # Read RGB image
        # color_image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        # print(color_image.shape, depth_im.shape)
        # color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)

        return cam_pose, depth_im, img_file


if __name__ == '__main__':
    scene_id = 'chess'
    tsdf = load_tsdf(scene_id, '/data/7scenes/tsdf')
    import ipdb; ipdb.set_trace()