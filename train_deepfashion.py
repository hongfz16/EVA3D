import os
import random
import yaml
import torch
import warnings
import numpy as np
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from typing import Optional

from torch.utils import data
from operator import itemgetter
from torch.nn import functional as F
from torch import nn, autograd, optim
from torchvision import transforms, utils
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, Sampler

from losses import *
from options import BaseOptions
from augment import AugmentPipe
from calculate_fid import get_fid
from dataset import DeepFashionDataset
from model import VolumeRenderDiscriminator
from model import VoxelHumanGenerator as Generator
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from utils import data_sampler, requires_grad, accumulate, sample_data, make_noise, mixing_noise, generate_camera_params

warnings.filterwarnings("ignore")

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

def train(opt, experiment_opt, _loader_dict, generator, discriminator, g_optim, d_optim, g_ema, device, augmentpipe):
    dataset, train_sampler = _loader_dict
    _loader = data.DataLoader(
        dataset,
        batch_size=opt.batch,
        sampler=train_sampler,
        drop_last=True,
    )
    loader = sample_data(_loader)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_eikonal = torch.tensor(0.0, device=device)
    g_minimal_surface = torch.tensor(0.0, device=device)

    g_loss_val = 0
    loss_dict = {}

    if opt.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = [torch.randn(opt.val_n_sample, opt.style_dim, device=device).repeat_interleave(8,dim=0)]
    sample_trans, sample_beta, sample_theta = _loader.dataset.sample_smpl_param(opt.val_n_sample, device)
    sample_cam_extrinsics, sample_focals = _loader.dataset.get_camera_extrinsics(opt.val_n_sample, device)

    pbar = range(opt.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    for idx in pbar:
        i = idx + opt.start_iter
        if i > opt.iter:
            print("Done!")
            break


        # Train discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        discriminator.zero_grad()
        _, real_imgs, cur_trans, cur_beta, cur_theta = next(loader)
        real_imgs = real_imgs.to(device)
        noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)

        cur_trans = cur_trans.to(device)
        cur_beta = cur_beta.to(device)
        cur_theta = cur_theta.to(device)
        cam_extrinsics, focal = _loader.dataset.get_camera_extrinsics(opt.batch, device)
        gen_imgs = []
        for j in range(0, opt.batch, opt.chunk):
            curr_noise = [n[j:j+opt.chunk] for n in noise]
            out = generator(curr_noise,
                            cam_extrinsics[j:j+opt.chunk],
                            focal[j:j+opt.chunk],
                            cur_beta[j:j+opt.chunk],
                            cur_theta[j:j+opt.chunk],
                            cur_trans[j:j+opt.chunk],
                            return_eikonal=False)

            gen_imgs += [out[1]]

        gen_imgs = torch.cat(gen_imgs, 0)
        fake_pred, _ = discriminator(augmentpipe(gen_imgs.detach().permute(0, 3, 1, 2).contiguous()))

        real_imgs.requires_grad = True
        real_pred, _ = discriminator(augmentpipe(real_imgs))
        d_gan_loss = d_logistic_loss(real_pred, fake_pred)
        grad_penalty = d_r1_loss(real_pred, real_imgs)
        r1_loss = opt.r1 * 0.5 * grad_penalty
        d_loss = d_gan_loss + r1_loss
        d_loss.backward()
        d_optim.step()

        loss_dict["d"] = d_gan_loss
        loss_dict["r1"] = r1_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()


        # Train Generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        for j in range(0, opt.batch, opt.chunk):
            noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
            _, _, cur_trans, cur_beta, cur_theta = next(loader)
            cur_trans = cur_trans.to(device)
            cur_beta = cur_beta.to(device)
            cur_theta = cur_theta.to(device)
            cam_extrinsics, focal = _loader.dataset.get_camera_extrinsics(opt.chunk, device)

            out = generator(noise, cam_extrinsics, focal, cur_beta, cur_theta, cur_trans,
                            return_sdf=opt.min_surf_lambda > 0,
                            return_eikonal=opt.eikonal_lambda > 0,
                            return_sdf_xyz=False)
            fake_img = out[1]
            if opt.min_surf_lambda > 0:
                sdf = out[2]
                if opt.eikonal_lambda > 0:
                    eikonal_term = out[3]
            elif opt.eikonal_lambda > 0:
                eikonal_term = out[2]

            fake_pred, _ = discriminator(augmentpipe(fake_img.permute(0, 3, 1, 2).contiguous()))

            if opt.with_sdf and opt.eikonal_lambda > 0:
                g_eikonal, g_minimal_surface = eikonal_loss(eikonal_term, sdf=sdf if opt.min_surf_lambda > 0 else None,
                                                            beta=opt.min_surf_beta, deltasdf=opt.deltasdf)
                g_eikonal = opt.eikonal_lambda * g_eikonal
                if opt.min_surf_lambda > 0:
                    g_minimal_surface = opt.min_surf_lambda * g_minimal_surface

            if opt.eikonal_lambda <= 0 and opt.min_surf_lambda > 0:
                g_minimal_surface = opt.min_surf_lambda * torch.exp(-opt.min_surf_beta * torch.abs(sdf)).mean()

            g_gan_loss = g_nonsaturating_loss(fake_pred)
            g_loss = g_gan_loss + g_eikonal + g_minimal_surface

            g_loss.backward()

        g_optim.step()
        generator.zero_grad()
        loss_dict["g"] = g_gan_loss
        loss_dict["g_eikonal"] = g_eikonal
        loss_dict["g_minimal_surface"] = g_minimal_surface

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        g_eikonal_loss = loss_reduced["g_eikonal"].mean().item()
        g_minimal_surface_loss = loss_reduced["g_minimal_surface"].mean().item()
        g_beta_val = g_module.renderer.sigmoid_beta.item() if opt.with_sdf else 0

        if opt.adjust_gamma:
            if opt.r1 >= opt.gamma_lb and i % 50000 == 0 and i != 0:
                opt.r1 = opt.r1 // 2

        if get_rank() == 0:
            pbar.set_description(
                (f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {opt.r1} {r1_val:.4f}; eik: {g_eikonal_loss:.4f}; surf: {g_minimal_surface_loss:.4f}; augp: {augmentpipe.p.item():.4f}; beta: {g_beta_val:.4f}")
            )

            if i % 100 == 0:
                with torch.no_grad():
                    samples = torch.Tensor(0, 3, opt.renderer_output_size[0], opt.renderer_output_size[1])
                    step_size = 1
                    mean_latent = g_module.mean_latent(10000, device)
                    for k in range(0, opt.val_n_sample, step_size):
                        out = g_ema([sample_z[0][k:k+step_size]],
                                                sample_cam_extrinsics[k:k+step_size],
                                                sample_focals[k:k+step_size],
                                                sample_beta[k:k+step_size],
                                                sample_theta[k:k+step_size],
                                                sample_trans[k:k+step_size],
                                                truncation=0.7,
                                                truncation_latent=mean_latent)
                        curr_samples = out[1]
                        samples = torch.cat([samples, curr_samples.cpu().permute(0, 3, 1, 2)[:, :3, ...]], 0)
                    samples = torch.cat([samples, augmentpipe(fake_img.permute(0, 3, 1, 2)[:, :3, ...]).cpu()], 0)
                    samples = torch.cat([samples, augmentpipe(real_imgs).cpu()[:, :3, ...]], 0)
                    if i % 100 == 0:
                        utils.save_image(samples,
                            os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', f"samples/{str(i).zfill(7)}.png"),
                            nrow=int(opt.val_n_sample),
                            normalize=True, range=(-1, 1))

            if i % 10000 == 0 or (i < 10000 and i % 1000 == 0):
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.freeze_renderer = False
    opt.training.camera = opt.camera
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.training.with_sdf = not opt.rendering.no_sdf
    if opt.training.with_sdf and opt.training.min_surf_lambda > 0:
        opt.rendering.return_sdf = True
    opt.rendering.no_features_output = True
    opt.training.sphere_init = False

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.training.distributed = n_gpu > 1

    if opt.training.distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # create checkpoints directories
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer'), exist_ok=True)
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', 'samples'), exist_ok=True)

    discriminator = VolumeRenderDiscriminator(opt.model).to(device)
    generator = Generator(opt.model, opt.rendering, full_pipeline=False, voxhuman_name=opt.model.voxhuman_name).to(device)
    g_ema = Generator(opt.model, opt.rendering, ema=True, full_pipeline=False, voxhuman_name=opt.model.voxhuman_name).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)
    g_optim = optim.Adam(generator.parameters(), lr=opt.training.glr, betas=(0, 0.9))
    d_optim = optim.Adam(discriminator.parameters(), lr=opt.training.dlr, betas=(0, 0.9))

    opt.training.start_iter = 0

    if opt.experiment.continue_training and opt.experiment.ckpt is not None:
        ckpt_path = os.path.join(opt.training.checkpoints_dir,
                                 opt.experiment.expname,
                                 'volume_renderer/models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        if get_rank() == 0:
            print("load model:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        try:
            opt.training.start_iter = int(opt.experiment.ckpt) + 1
        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=True)
        discriminator.load_state_dict(ckpt["d"], strict=True)
        g_ema.load_state_dict(ckpt["g_ema"])
        if "g_optim" in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])

    #AugmentPipe
    if opt.training.small_aug:
        scale_std = 0.05
        bgc_dict = dict(xint=1, xint_max=0.05, scale=1, scale_std=scale_std, rotate=1, rotate_max=0.025)
        augmentpipe = AugmentPipe(**bgc_dict).train().requires_grad_(False).to(device)
        augmentpipe.p.copy_(torch.as_tensor(0.6))
    else:
        bgc_dict = dict(xflip=1, xint=1, scale=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        augmentpipe = AugmentPipe(**bgc_dict).train().requires_grad_(False).to(device)
        augmentpipe.p.copy_(torch.as_tensor(0))

    if opt.training.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[opt.training.local_rank],
            output_device=opt.training.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[opt.training.local_rank],
            output_device=opt.training.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    dataset = DeepFashionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                opt.model.renderer_spatial_output_dim,
                                os.path.join(opt.dataset.dataset_path, 'train_list.txt'),
                                white_bg=opt.rendering.white_bg,
                                random_flip=opt.dataset.random_flip,
                                gaussian_weighted_sampler=opt.dataset.gaussian_weighted_sampler,
                                sampler_std=opt.dataset.sampler_std)

    if opt.dataset.gaussian_weighted_sampler:
        sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights))
        if opt.training.distributed:
            train_sampler = DistributedSamplerWrapper(sampler)
        else:
            train_sampler = sampler
    else:
        train_sampler = data_sampler(dataset, shuffle=True, distributed=opt.training.distributed)

    opt.training.dataset_name = opt.dataset.dataset_path.lower()

    # save options
    opt_path = os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', f"opt.yaml")
    with open(opt_path,'w') as f:
        yaml.safe_dump(opt, f)

    train(opt.training, opt.experiment, (dataset, train_sampler), generator, discriminator, \
          g_optim, d_optim, g_ema, device, augmentpipe)
