import math
import torch
from torch import autograd
from torch.nn import functional as F


def viewpoints_loss(viewpoint_pred, viewpoint_target):
    loss = F.smooth_l1_loss(viewpoint_pred, viewpoint_target)

    return loss


def eikonal_loss(eikonal_term, sdf=None, beta=100, deltasdf=False):
    if eikonal_term == None:
        eikonal_loss = 0
    else:
        if deltasdf:
            eikonal_loss = ((eikonal_term.norm(dim=-1)) ** 2).mean()
        else:
            eikonal_loss = ((eikonal_term.norm(dim=-1) - 1) ** 2).mean()

    if sdf == None:
        minimal_surface_loss = torch.tensor(0.0, device=eikonal_term.device)
    else:
        if deltasdf:
            minimal_surface_loss = torch.nn.functional.smooth_l1_loss(sdf, torch.zeros_like(sdf)).mean()
        else:
            minimal_surface_loss = torch.exp(-beta * torch.abs(sdf)).mean()

    return eikonal_loss, minimal_surface_loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(),
                               inputs=real_img,
                               create_graph=True)

    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents,
                         create_graph=True, only_inputs=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths
