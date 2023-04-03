import cv2
import numpy as np
from torch.utils.data import Dataset


ROOT = '/Users/gongxuan/Desktop/data'
fx, fy = 767.3861511125845, 767.5058656118406
cx, cy = 679.054265997005, 543.646891684636
distCoef = [-0.18867185058223412,-0.003927337093919806,0.030524814153620117,-0.012756926010904904]


class Colon3DVDataset(Dataset):
    def __init__(self, root_path=ROOT, video='c1v1' ,selectn=None):
        pose_file = f"{root_path}/{video}/pose.txt"
        with open(pose_file) as f:
            lines = f.readlines()
        poses = []
        image_files = []
        depth_files = []
        for n, line in enumerate(lines):
            line = line.replace("\n", "").split(',')
            assert len(line)==16
            pose = np.array([float(l) for l in line]).reshape((4,4))
            poses.append(pose)
            image_files.append(f"{root_path}/{video}/{n}_color.png")
            depth_files.append(f"{root_path}/{video}/{n}_depth.tiff")
        self.poses = np.stack(poses, 0)
        self.imagef = np.stack(image_files, 0)
        self.depthf = np.stack(depth_files, 0)

        #select subset
        if selectn is not None:
            indices = np.arange(len(poses)).tolist()
            freq = len(poses)//selectn
            select_indices = indices[::freq]
            self.poses = self.poses[select_indices]
            self.imagef = self.imagef[select_indices]
            self.depthf = self.depthf[select_indices]
        #

        self.focal = np.array([fx, fy]).astype(np.float32)
        self.principal = np.array([cx, cy]).astype(np.float32)
        self.distcoef = np.array(distCoef).astype(np.float32)

    def __len__(self):
        return len(self.poses)



    def __getitem__(self, idx):
        pose = self.poses[idx].astype(np.float32)
        image = cv2.imread(self.imagef[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depthf[idx],3)[:,:,0]/65535. #mm
        depth = depth.astype(np.float32) *100
        
        # return {"image": image,
        #         "imagef": self.imagef[idx],
        #         "depth": depth,
        #         "c2w": pose}
        return pose, depth, self.imagef[idx]