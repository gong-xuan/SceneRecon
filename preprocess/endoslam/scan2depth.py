import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import sys
sys.path.append("../../")
from ops.p3drender import Renderer_Pytorch3D
import trimesh
import pandas
from scipy.spatial.transform import Rotation

#Colon-IV
root_path = '/data_local/endo_datasets/endoslam_raw/Cameras/HighCam/Colon-IV/TumorfreeTrajectory_1'
# intr_path = '/data_local/endo_datasets/endoslam_raw/Cameras/HighCam/Calibration/cam.txt.txt'
high_intrinsincs = np.array([[957.411, 5.6242, 282.192],
                            [0, 959.386, 170.731],
                            [0, 0, 1]])

class EndoSlamDataset(Dataset):
    def __init__(self):
        pose_file = f"{root_path}/Poses/low_high_pose_colon_test_1_high_images.xlsx"
        pose_data = pandas.read_excel(pose_file)
        self.pose_values = pose_data.values
        self.K = high_intrinsincs.astype(np.float32)

    def __len__(self):
        return self.pose_values.shape[0]

    def __getitem__(self, idx):
        frame = self.pose_values[idx,1].astype('int')
        image = cv2.imread(f"{root_path}/Frames/frame_{frame:06d}.jpg")
        trans = self.pose_values[idx, 3:6].astype(np.float32)
        quat = self.pose_values[idx, 6:10].astype(np.float32)
        rot = Rotation.from_quat(quat).as_matrix()
        
        return {"images": image,
                "R": rot,
                "T": trans,
                "K": self.K}

    # pose_file = f"{root_path}/Poses/low_high_pose_colon_test_1_high_images.xlsx"
    # pose_data = pandas.read_excel(pose_file)
    # frames = pose_data.values[:,1].astype('int').tolist()
    # for frame in frames:
    #     imgpath = f"{root_path}/Frames/frame_{frame:06d}.jpg"
    #     assert os.path.exists(imgpath)
    #     image = cv2.imread(imgpath)
    #     print(image.shape)

if __name__ == '__main__':
    H, W, batch_size = 480, 640, 1
    dataset = EndoSlamDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=batch_size)

    mesh_file = "/data_local/endo_datasets/endoslam_raw/scan3d/Colon-IV/Colon-IV.ply"
    mesh = trimesh.load(mesh_file, process=False)#vertices and faces
    renderer = Renderer_Pytorch3D(batch_size, 
                                device='cuda:1', 
                                height=H, 
                                width=W)
    renderer.set_intrinsics_mesh(high_intrinsincs, mesh) # not use skew in intrinsics
    # R,T 
    for samples in dataloader:
        R = samples["R"]
        T = samples["T"]
        # w2c = torch.eye(4)[None].repeat(batch_size, 1, 1)
        # T->R, world2cam, R->T, world2cam
        w2c_1 = torch.eye(4)[None].repeat(batch_size, 1, 1)
        w2c_1[:,:3,:3] = R
        w2c_2 =  torch.eye(4)[None].repeat(batch_size, 1, 1)
        w2c_2[:,:3,3] = T
        w2c = w2c_1@w2c_2
        import ipdb; ipdb.set_trace()

        depth = renderer(w2c.transpose(1,2), world2cam=True)
    # 
    # import ipdb; ipdb.set_trace()
    
    