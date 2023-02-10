import torch
import random
import trimesh
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
from pdb import set_trace as st

import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)

######################### Dataset util functions ###########################
# Get data sampler
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

# Get data minibatch
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

############################## Model weights util functions #################
# Turn model gradients on/off
def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

################### Latent code (Z) sampling util functions ####################
# Sample Z space latent codes for the generator
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]

################# Camera parameters sampling ####################
def generate_camera_params(resolution, device, batch=1, locations=None, sweep=False,
                           uniform=False, azim_range=0.3, elev_range=0.15,
                           fov_ang=6, dist_radius=0.12):
    if locations != None:
        azim = locations[:,0].view(-1,1)
        elev = locations[:,1].view(-1,1)

        # generate intrinsic parameters
        # fix distance to 1
        dist = torch.ones(azim.shape[0], 1, device=device)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(azim.shape[0], 1, device=device).view(-1,1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
        # generate camera locations on the unit sphere
        azim = (-azim_range + (2 * azim_range / 7) * torch.arange(8, device=device)).view(-1,1).repeat(batch,1)
        elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device).repeat(1,8).view(-1,1))

        # generate intrinsic parameters
        dist = (torch.ones(batch, 1, device=device)).repeat(1,8).view(-1,1)
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(batch, 1, device=device).repeat(1,8).view(-1,1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    else:
        # sample camera locations on the unit sphere
        if uniform:
            azim = (-azim_range + 2 * azim_range * torch.rand(batch, 1, device=device))
            elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device))
        else:
            azim = (azim_range * torch.randn(batch, 1, device=device))
            elev = (elev_range * torch.randn(batch, 1, device=device))

        # generate intrinsic parameters
        dist = torch.ones(batch, 1, device=device) # restrict camera position to be on the unit sphere
        near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(batch, 1, device=device) * np.pi / 180 # full fov is 12 degrees
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1,3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0,1,0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir, eps=1e-5) # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1,2),T), -1)

    return extrinsics, focal, near, far, viewpoint

#################### Mesh generation util functions ########################
# Reshape sampling volume to camera frostum
def align_volume(volume, near=0.88, far=1.12):
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w),
                                torch.linspace(-1, 1, d))

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = torch.linspace(far / near, 1, d).view(1,1,1,-1,1).to(volume.device)
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[...,:2] = frostum_grid[...,:2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any((frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True)
    frostum_grid = frostum_grid.permute(0,3,1,2,4).contiguous()
    permuted_volume = volume.permute(0,4,3,1,2).contiguous()
    final_volume = F.grid_sample(permuted_volume, frostum_grid, padding_mode="border", align_corners=True)
    final_volume = final_volume.permute(0,3,4,2,1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 1

    return final_volume

# Extract mesh with marching cubes
def extract_mesh_with_marching_cubes(sdf, level_set=0):
    # b, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    # sdf_vol = sdf[0,...,0].permute(1,0,2).cpu().numpy()


    w, h, d = sdf.shape
    sdf_vol = sdf.cpu().numpy()

    # scale vertices
    verts, faces, _, _ = marching_cubes(sdf_vol, level_set, mask=sdf_vol!=10086)
    verts[:,0] = (verts[:,0]/float(w)-0.5)
    verts[:,1] = (verts[:,1]/float(h)-0.5)
    verts[:,2] = (verts[:,2]/float(d)-0.5)

    # # fix normal direction
    verts[:,2] *= -1; verts[:,1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh, verts, faces

# Generate mesh from xyz point cloud
def xyz2mesh(xyz):
    xyz = xyz.permute(0, 3, 1, 2)
    b, _, h, w = xyz.shape
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    # Extract mesh faces from xyz maps
    tri = Delaunay(np.concatenate((x.reshape((h*w, 1)), y.reshape((h*w, 1))), 1))
    faces = tri.simplices

    # invert normals
    faces[:,[0, 1]] = faces[:,[1, 0]]

    # generate_meshes
    mesh = trimesh.Trimesh(xyz.squeeze(0).permute(1,2,0).view(h*w,3).cpu().numpy(), faces)

    return mesh


################# Mesh rendering util functions #############################
def add_textures(meshes:Meshes, vertex_colors=None) -> Meshes:
    verts = meshes.verts_padded()
    if vertex_colors is None:
        vertex_colors = torch.ones_like(verts) # (N, V, 3)
    textures = TexturesVertex(verts_features=vertex_colors)
    meshes_t = Meshes(
        verts=verts,
        faces=meshes.faces_padded(),
        textures=textures,
        verts_normals=meshes.verts_normals_padded(),
    )
    return meshes_t


def create_cameras(
    R=None, T=None,
    azim=0, elev=0., dist=1.,
    fov=12., znear=0.01,
    device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=fov)
    return cameras


def create_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0),),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=light_location, **light_kwargs
    )
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return phong_renderer


## custom renderer
class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def create_depth_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=17,
    )
    # We can add a point light in front of the object.
    lights = PointLights(
        device=device, location=((-0.5, 1., 5.0),), **light_kwargs
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
            device=device,
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return renderer
