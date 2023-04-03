import numpy as np
import torch


def undistort_cam_pts_np(pts2d, distCoef):
    #pts2d: [2, N]
    N_coef = distCoef.shape[0]
    k = np.zeros(12)
    k[:N_coef] = distCoef

    x0, y0 = pts2d[0,:], pts2d[1,:]
    x, y = x0.copy(), y0.copy()
    for _ in range(N_coef):
        r2 = x*x + y*y
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
        deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist
    return np.stack([x,y])

def distort_cam_pts_np(pts3d, distCoef):
    #pts3d: [3, N]
    if pts3d.shape[0]==3:
        x0, y0, z0 = pts3d[0,:], pts3d[1,:], pts3d[2,:]
        zero_mask = (z0==0)
        a, b = x0[~zero_mask]/z0[~zero_mask], y0[~zero_mask]/z0[~zero_mask]
    else:
        a, b = pts3d[0,:], pts3d[1,:]
        zero_mask = None
    
    r = np.sqrt(a*a + b*b)
    theta = np.arctan(r)
    k = distCoef[:4]
    theta_d = theta*(1 + k[0]*np.power(theta, 2) + k[1]*np.power(theta, 4) + k[2]*np.power(theta, 6) + k[1]*np.power(theta, 8))

    x = a*theta_d/r
    y = b*theta_d/r

    if zero_mask is None:
        return np.stack([x,y])
    else:
        x0[~zero_mask] = x*z0[~zero_mask]
        y0[~zero_mask] = y*z0[~zero_mask]
        return np.stack([x0,y0,z0])


def distort_cam_pts_torch(pts3d, distCoef):
    #pts3d: [N, 3]
    if pts3d.shape[1]==3:
        x0, y0, z0 = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
        zero_mask = (z0==0)
        a, b = x0[~zero_mask]/z0[~zero_mask], y0[~zero_mask]/z0[~zero_mask]
    else:
        a, b = pts3d[:, 0], pts3d[:, 1]
        zero_mask = None
    
    r = torch.sqrt(a*a + b*b)
    theta = torch.arctan(r)
    k = distCoef[:4]
    theta_d = theta*(1 + k[0]*torch.pow(theta, 2) + k[1]*torch.pow(theta, 4) + k[2]*torch.pow(theta, 6) + k[1]*torch.pow(theta, 8))

    x = a*theta_d/r
    y = b*theta_d/r

    if zero_mask is None:
        return torch.stack([x,y],dim=1)
    else:
        x0[~zero_mask] = x*z0[~zero_mask]
        y0[~zero_mask] = y*z0[~zero_mask]
        return torch.stack([x0,y0,z0], dim=1)