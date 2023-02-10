import os
import torch
import trimesh
import imageio
import pickle
import numpy as np
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms
from skimage.measure import marching_cubes
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from options import BaseOptions
from model import VoxelHumanGenerator as Generator
from dataset import DeepFashionDataset, DemoDataset
from utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
    requires_grad,
    create_mesh_renderer,
    create_cameras
)
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

# torch.random.manual_seed(10086)
# import random
# random.seed(10086)

panning_angle = np.pi / 3

def generate(opt, dataset, g_ema, device, mean_latent, is_video):
    requires_grad(g_ema, False)
    g_ema.is_train = False
    g_ema.train_renderer = False
    sample_z_list = {}
    for i in tqdm(range(opt.identities)):
        if is_video:
            sample_z = torch.randn(1, opt.style_dim, device=device)
        else:
            # if i % 2 == 0:
            sample_z = torch.randn(1, opt.style_dim, device=device)
        sample_z_list[str(i).zfill(7)] = sample_z.cpu().numpy()
        sample_trans, sample_beta, sample_theta = dataset.sample_smpl_param(1, device, val=False)
        sample_cam_extrinsics, sample_focals = dataset.get_camera_extrinsics(1, device, val=False)

        if is_video:
            video_list = []
            for k in tqdm(range(120)):
                if k < 30:
                    angle = (panning_angle / 2) * (k / 30)
                elif k >= 30 and k < 90:
                    angle = panning_angle / 2 - panning_angle * ((k - 30) / 60)
                else:
                    angle = -panning_angle / 2 * ((120 - k) / 30)
                delta = R.from_rotvec(angle * np.array([0, 1, 0]))
                r = R.from_rotvec(sample_theta[0, :3].cpu().numpy())
                new_r = delta * r
                new_sample_theta = sample_theta.clone()
                new_sample_theta[0, :3] = torch.from_numpy(new_r.as_rotvec()).to(device)
                with torch.no_grad():
                    j = 0
                    chunk = 1
                    out = g_ema([sample_z[j:j+chunk]],
                                sample_cam_extrinsics[j:j+chunk],
                                sample_focals[j:j+chunk],
                                sample_beta[j:j+chunk],
                                new_sample_theta[j:j+chunk],
                                sample_trans[j:j+chunk],
                                truncation=opt.truncation_ratio,
                                truncation_latent=mean_latent,
                                return_eikonal=False,
                                return_normal=False,
                                return_mask=False,
                                fix_viewdir=True)
                rgb_images_thumbs = out[1].detach().cpu()[..., :3]
                g_ema.zero_grad()
                video_list.append((rgb_images_thumbs.numpy() + 1) / 2. * 255. + 0.5)
            all_img = np.concatenate(video_list, 0).astype(np.uint8)
            imageio.mimwrite(os.path.join(opt.results_dst_dir, 'images_paper_video', 'video_{}.mp4'.format(str(i).zfill(7))), all_img, fps=30, quality=8)
        else:
            img_list = []
            for k in range(3):
                if k == 0:
                    delta = R.from_rotvec(np.pi/8 * np.array([0, 1, 0]))
                elif k == 2:
                    delta = R.from_rotvec(-np.pi/8 * np.array([0, 1, 0]))
                else:
                    delta = R.from_rotvec(0 * np.array([0, 1, 0]))
                r = R.from_rotvec(sample_theta[0, :3].cpu().numpy())
                new_r = delta * r
                new_sample_theta = sample_theta.clone()
                new_sample_theta[0, :3] = torch.from_numpy(new_r.as_rotvec()).to(device)

                with torch.no_grad():
                    j = 0
                    chunk = 1
                    out = g_ema([sample_z[j:j+chunk]],
                                sample_cam_extrinsics[j:j+chunk],
                                sample_focals[j:j+chunk],
                                sample_beta[j:j+chunk],
                                new_sample_theta[j:j+chunk],
                                sample_trans[j:j+chunk],
                                truncation=opt.truncation_ratio,
                                truncation_latent=mean_latent,
                                return_eikonal=False,
                                return_normal=False,
                                return_mask=False,
                                fix_viewdir=True)

                rgb_images_thumbs = out[1].detach().cpu()[..., :3].permute(0, 3, 1, 2)
                g_ema.zero_grad()
                img_list.append(rgb_images_thumbs)
        ##################################
        latent = g_ema.styles_and_noise_forward(sample_z[:1], None, opt.truncation_ratio,
                                                mean_latent, False)

        sdf = g_ema.renderer.marching_cube_posed(latent[0], sample_beta, sample_theta, resolution=500, size=1.4).detach()
        marching_cubes_mesh, _, _ = extract_mesh_with_marching_cubes(sdf, level_set=0)
        marching_cubes_mesh = trimesh.smoothing.filter_humphrey(marching_cubes_mesh, beta=0.2, iterations=5)
        marching_cubes_mesh_filename = os.path.join(opt.results_dst_dir,'marching_cubes_meshes_posed','sample_{}_marching_cubes_mesh.obj'.format(i))
        with open(marching_cubes_mesh_filename, 'w') as f:
            marching_cubes_mesh.export(f,file_type='obj')
        ##################################
        cam_R = torch.eye(3).cuda().unsqueeze(0)
        cam_R[0, 2, 2] = -1
        cam_R[0, 0, 0] = -1
        cam_trans = sample_trans.view(1, -1)
        cam_trans[0, :2] = 0
        # if '20w_fashion' in dataset.path:
        #     cam_trans /= 2.
        camera = create_cameras(R = cam_R, T = cam_trans, fov=4.5)
        # renderer = create_mesh_renderer(camera, 256)
        renderer = create_mesh_renderer(camera, opt.renderer_output_size[0])
        verts, faces_idx, _ = load_obj(marching_cubes_mesh_filename)
        faces = faces_idx.verts_idx
        verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        if is_video:
            video_list = []
            for k in tqdm(range(120)):
                verts_clone = verts.clone().cpu().numpy()
                if k < 30:
                    angle = (-panning_angle/2) * (k / 30)
                elif k >= 30 and k < 90:
                    angle = -panning_angle/2 + panning_angle * ((k - 30) / 60)
                else:
                    angle = panning_angle/2 * ((120 - k) / 30)
                delta = R.from_rotvec(angle * np.array([0, 1, 0]))
                verts_clone = torch.from_numpy(delta.apply(verts_clone)).float()
                pt3d_mesh = Meshes(
                    verts=[verts_clone.to(device)],
                    faces=[faces.to(device)],
                    textures=textures
                )
                image = (renderer(pt3d_mesh) * 255. + 0.5)[:, :, :, :3].cpu().numpy()
                video_list.append(image)
            all_img = np.concatenate(video_list, 0).astype(np.uint8)
            imageio.mimwrite(os.path.join(opt.results_dst_dir, 'images_paper_video', 'geo_{}.mp4'.format(str(i).zfill(7))), all_img, fps=30, quality=8)
        else:
            for k in range(3):
                verts_clone = verts.clone().cpu().numpy()
                if k == 0:
                    delta = R.from_rotvec(-np.pi/8 * np.array([0, 1, 0]))
                elif k == 2:
                    delta = R.from_rotvec(np.pi/8 * np.array([0, 1, 0]))
                else:
                    delta = R.from_rotvec(0 * np.array([0, 1, 0]))
                verts_clone = torch.from_numpy(delta.apply(verts_clone)).float()
                pt3d_mesh = Meshes(
                    verts=[verts_clone.to(device)],
                    faces=[faces.to(device)],
                    textures=textures
                )
                image = renderer(pt3d_mesh)
                image = image * 2 - 1
                img_list.append(image.reshape(opt.renderer_output_size[0], opt.renderer_output_size[0], 4)[:, opt.renderer_output_size[0]//4:opt.renderer_output_size[0]//4*3, :3].unsqueeze(0).permute(0, 3, 1, 2).cpu())

            utils.save_image(torch.cat(img_list, 0),
                os.path.join(opt.results_dst_dir, 'images_paper_fig','{}.png'.format(str(i).zfill(7))),
                nrow=3,
                normalize=True,
                range=(-1, 1),
                padding=0,)
        # os.system('rm {}'.format(marching_cubes_mesh_filename))

if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.freeze_renderer = False
    opt.rendering.no_features_output = True
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz
    
    checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname, 'volume_renderer')
    checkpoint_path = os.path.join(checkpoints_dir,
                                    'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
    # define results directory name
    result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    # create results directory
    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
    if opt.inference.fixed_camera_angles:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'fixed_angles')
    else:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'random_angles')
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    if not opt.rendering.render_video:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images_paper_fig'), exist_ok=True)
    else:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images_paper_video'), exist_ok=True)
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'marching_cubes_meshes_posed'), exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    # load generation model
    g_ema = Generator(opt.model, opt.rendering, full_pipeline=False, voxhuman_name=opt.model.voxhuman_name).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v
        else:
            print(k)

    g_ema.load_state_dict(model_dict)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    
    if 'deepfashion' in opt.dataset.dataset_path:
        file_list = '/mnt/lustre/fzhong/smplify-x/deepfashion_train_list/deepfashion_train_list_MAN.txt'
    elif '20w_fashion' in opt.dataset.dataset_path:
        file_list = '/mnt/lustre/fzhong/mmhuman3d/20w_fashion_result/nondress_flist.txt'
    else:
        file_list = None
    if file_list:
        dataset = DeepFashionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                     opt.model.renderer_spatial_output_dim, file_list)
    else:
        dataset = DemoDataset()

    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        mean_latent = None

    g_ema.renderer.is_train = False
    g_ema.renderer.perturb = 0

    generate(opt.inference, dataset, g_ema, device, mean_latent, opt.rendering.render_video)
