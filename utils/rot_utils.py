import numpy as np
import torch

rot_x = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),np.sin(phi),0],
    [0,-np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_y = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_z = lambda th : torch.Tensor([
    [np.cos(th),np.sin(th), 0, 0],
    [-np.sin(th),np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0,0,0,1]]).float()