import torch
from typing import NamedTuple, Sequence
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PointLights, HardPhongShader

from utils.utils import make_nograd_func
from utils.distort_utils import distort_cam_pts_torch
from pytorch3d.ops import interpolate_face_attributes


def persepctive_project(points, cam_R, cam_T=None, cam_K=None, rot_first=True):
    """
    This function computes the perspective projection of a set of points in torch.
    Input:
        points (bs, N, 3): 3D points
        cam_R (bs, 3, 3): Camera rotation
        cam_T (bs, 3): Camera translation
        cam_K (bs, 3, 3): Camera intrinsics matrix
        
    """
    batch_size = points.shape[0]

    # Transform points
    # import ipdb; ipdb.set_trace()
    if rot_first:
        points = torch.einsum('bij,bkj->bki', cam_R, points)
    if cam_T is not None:
        points = points + cam_T.unsqueeze(1)
    if not rot_first:
        points = torch.einsum('bij,bkj->bki', cam_R, points)
    if cam_K is None:
        return points
    else:
        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', cam_K, projected_points)
        # 
        return projected_points[:, :, :-1]


def DynamicRenderer_Pytorch3D(vertices_list, faces_list, cam_intrinsics, cam_extrinsics, image_size, world2cam, device):
    """
    vertices_list: (bs, Nv, 3)
    faces_list: (bs, Nface, 3)
    cam_intrinsics: (bs, nview, 3, 3) -> flattend (bs*nview)
    cam_extrinsics: (bs, nview, 4, 4) -> flattend (bs*nview)
    world2cam: extrinsics is world2cam or cam2world
    """
    bs, nview = cam_intrinsics.shape[:2]
    #expand to multiviews
    # vertices_list = [verts for verts in vertices_list for _ in range(nview)] 
    # #duplicate nview times for each sample
    faces_list = [faces for faces in faces_list for _ in range(nview)]
    #project verts
    cam_R = cam_extrinsics[:,:, :3,:3]
    cam_T = cam_extrinsics[:,:, :3, 3]
    if world2cam:
        rot_first = True
    else:
        rot_first = False
        cam_R = cam_R.transpose(2, 3)
        cam_T = cam_T*(-1) 
    
    verts_list = []
    for b, vertices in enumerate(vertices_list):
        # extrinsics = cam_extrinsics[n] #(nview, 4, 4)
        vertices = vertices[None].expand(nview, -1, -1)
        verts = persepctive_project(vertices, 
                                    cam_R[b], 
                                    cam_T = cam_T[b], 
                                    cam_K = None,
                                    rot_first = rot_first)
        for nv in range(nview):
            verts_list.append(verts[nv])

    #
    cam_intrinsics = cam_intrinsics.reshape(-1, 3, 3)
    BS = cam_intrinsics.shape[0]

    fx = cam_intrinsics[:,0,0]
    fy = cam_intrinsics[:,1,1]
    tx = cam_intrinsics[:,0,2]
    ty = cam_intrinsics[:,1,2]
    focal_length = torch.stack([fx,fy], dim=1)
    principal_point = torch.stack([tx, ty], dim=1)
    calibrate_R = torch.tensor([[-1., 0., 0.], 
                                [0., -1., 0.], 
                                [0., 0., 1.]]).to(device)[None].expand(BS, -1, -1)
    
    cameras = PerspectiveCameras(
            focal_length = focal_length, # A tensor of shape (N, 1) or (N, 2) 
            principal_point = principal_point, #A tensor of shape (N, 2)
            K = None,
            R =calibrate_R,#tensor(bs,3,3)
            # T = cam_T,#tensor(bs,3)
            device = device, 
            in_ndc = False,
            image_size = (image_size,)
        )
    #Bin size was too small in the coarse rasterization phase.
    # This caused an overflow, meaning output may be incomplete. 
    # To solve, try increasing max_faces_per_bin / max_points_per_bin, decreasing bin_size, 
    # or setting bin_size to 0 to use the naive rasterization.

    raster_settings = RasterizationSettings(
                bin_size = 0,
                image_size = image_size,
                blur_radius=0.0,
                faces_per_pixel=1, #topK value
            )
    rasterizer = MeshRasterizer(cameras=cameras,
                                raster_settings = raster_settings)
    
    # import ipdb; ipdb.set_trace()
    mesh_batch = Meshes(verts = verts_list, faces = faces_list, textures = None)
    fragments = rasterizer(mesh_batch)
    depth = fragments.zbuf.squeeze(dim=-1) #(bs*nview, imgh, imgw)
    depth = depth.reshape(bs, nview, *image_size)
    # import ipdb; ipdb.set_trace()
    return depth




class BlendParams(NamedTuple):
            sigma: float = 1e-4
            gamma: float = 1e-4
            background_color: Sequence = (0.0, 0.0, 0.0)


        
class Renderer_Pytorch3D:
    def __init__(self, batch_size, device='cpu', height=480, width=640):
        """
        OpenGL mesh renderer
        Used to render depthmaps from a mesh for 2d evaluation
        """
        self.height = height
        self.width = width
        self.device = device
        self.batch_size = batch_size
        self.ts_type = torch.float32

        self.raster_settings = RasterizationSettings(
            image_size = (height, width ),
            blur_radius=0.0,
            faces_per_pixel=1, #topK value
        )
        # for rgb renderer
        self.lights_rgb_render = PointLights(device=self.device,
                                    location=((0.0, 0.0, -2.0),),
                                    ambient_color=((0.5, 0.5, 0.5),),
                                    diffuse_color=((0.3, 0.3, 0.3),),
                                    specular_color=((0.2, 0.2, 0.2),))
        self.blendparam = BlendParams()
    
    def set_intrinsics_mesh(self, cam_intr, mesh, verts_rgb_colors=None, distCoef=None):
        #intrinsic
        fx, fy, tx, ty = cam_intr[0,0], cam_intr[1,1], cam_intr[0,2], cam_intr[1,2]
        focal_length = [fx,fy]
        focal_length = torch.tensor(focal_length)[None].to(self.ts_type).to(self.device)
        #cameras
        calibrate_R = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).to(self.device)[None]#.expand(batch_size, -1, -1)
        self.cameras = PerspectiveCameras(
            focal_length = focal_length,
            principal_point = ((tx, ty),),
            K = None,
            R =calibrate_R,#tensor(bs,3,3)
            # T = cam_T,#tensor(bs,3)
            device = self.device, 
            in_ndc = False,
            image_size = ((self.height, self.width),)
        )
        self.rasterizer = MeshRasterizer(cameras=self.cameras,
                                raster_settings = self.raster_settings)


        #mesh
        self.faces_list = [torch.tensor(mesh.faces).to(self.ts_type).to(self.device) for _ in range(self.batch_size)]
        self.vertices_ts = torch.tensor(mesh.vertices).to(self.ts_type).to(self.device)[None].expand(self.batch_size, -1, -1)
        if verts_rgb_colors is not None:
            self.textures = Textures(verts_rgb=torch.tensor(verts_rgb_colors).to(self.ts_type).to(self.device)[None].expand(self.batch_size, -1, -1))
        else:
            self.textures = None

        
        self.distCoef = distCoef
        

    @make_nograd_func
    def __call__(self, cam_pose, world2cam=False, cameras=None, render_rgb=False, render_normal=False, dist=False):
        """
        #cam_pose: torchtensor (bs, 4, 4)
        # if cam_pose is cam2world, convert to world2cam
        """
        cam_pose = cam_pose.to(self.ts_type).to(self.device)
        cam_R = cam_pose[:, :3, :3]
        cam_T = cam_pose[:, :3, 3]
        if world2cam:
            rot_first = True
        else:
            rot_first = False
            cam_R = cam_R.transpose(1,2)
            cam_T = cam_T*(-1) 
        #
        current_bs = cam_pose.size(0)
        assert current_bs<= self.batch_size
        if current_bs!=self.batch_size:
            faces_list = self.faces_list[:current_bs]  
            vertices = self.vertices_ts[:current_bs].clone()
        else:
            faces_list = self.faces_list 
            vertices = self.vertices_ts.clone()

        vertices = persepctive_project(vertices, 
                                        cam_R, 
                                        cam_T = cam_T, 
                                        cam_K = None,
                                        rot_first = rot_first)#w2c
        if self.distCoef is not None and dist:
            vertices = distort_cam_pts_torch(vertices.reshape((-1,3)), self.distCoef).reshape((current_bs, -1, 3))

        verts_list = [vertices[batch] for batch in range(current_bs)]

        mesh_batch = Meshes(verts = verts_list,
                            faces = faces_list,
                            textures = self.textures)
        if cameras is not None:
            raster_settings = RasterizationSettings(
                image_size = cameras.image_size,
                blur_radius=0.0,
                faces_per_pixel=1, #topK value
            )
            rasterizer = MeshRasterizer(cameras=cameras,
                                raster_settings = raster_settings)
        else:
            rasterizer = self.rasterizer
        
        fragments = rasterizer(mesh_batch)
        depth = fragments.zbuf.squeeze(dim=-1)

        rgb_images, normals = None, None
        if render_rgb:
            rgb_shader = HardPhongShader(device=self.device, cameras=cameras if cameras is not None else self.cameras,
                                        lights=self.lights_rgb_render, blend_params=self.blendparam)
            rgb_images = rgb_shader(fragments, mesh_batch, lights=self.lights_rgb_render)[:, :, :, :3]
            rgb_images = torch.clamp(rgb_images, max=1.0)
        
        if render_normal:
            normals = self.get_pixel_normals(mesh_batch, fragments)
        return depth, rgb_images, normals

    def get_pixel_normals(self, mesh_batch, fragments):
        vertex_normals = mesh_batch.verts_normals_packed() #1096060=140*7829
        faces = mesh_batch.faces_packed()  # (bs*F, 3)
        faces_normals = vertex_normals[faces]

        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )
        return pixel_normals.squeeze(dim=-2)