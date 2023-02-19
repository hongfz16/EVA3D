import math
import random
import trimesh
import torch
import imageio
import numpy as np
from torch import nn
from torch.nn import functional as F
from volume_renderer import VolumeFeatureRenderer
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from pdb import set_trace as st

from eva3d_deepfashion import VoxelHuman as EVA3D_DEEPFASHION_MODEL
from eva3d_aist import VoxelHuman as EVA3D_AIST_MODEL

# import neural_renderer as nr
from smpl_utils import batch_rodrigues

from utils import (
    create_cameras,
    create_mesh_renderer,
    add_textures,
    create_depth_mesh_renderer,
)
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.transforms import matrix_to_euler_angles


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MappingLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, is_last=False):
        super().__init__()
        if is_last:
            weight_std = 0.25
        else:
            weight_std = 1

        self.weight = nn.Parameter(weight_std * nn.init.kaiming_normal_(torch.empty(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        if bias:
            self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):
        if self.activation != None:
            out = F.linear(input, self.weight)
            out = fused_leaky_relu(out, self.bias, scale=1)
        else:
            out = F.linear(input, self.weight, bias=self.bias)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale,
                           bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self, project=False):
        super().__init__()
        self.project = project
        self.weight = nn.Parameter(torch.zeros(1))
        self.prev_noise = None
        self.mesh_fn = None
        self.vert_noise = None

    def create_pytorch_mesh(self, trimesh):
        v=trimesh.vertices; f=trimesh.faces
        verts = torch.from_numpy(np.asarray(v)).to(torch.float32).cuda()
        mesh_pytorch = Meshes(
            verts=[verts],
            faces = [torch.from_numpy(np.asarray(f)).to(torch.float32).cuda()],
            textures=None
        )
        if self.vert_noise == None or verts.shape[0] != self.vert_noise.shape[1]:
            self.vert_noise = torch.ones_like(verts)[:,0:1].cpu().normal_().expand(-1,3).unsqueeze(0)

        mesh_pytorch = add_textures(meshes=mesh_pytorch, vertex_colors=self.vert_noise.to(verts.device))

        return mesh_pytorch

    def load_mc_mesh(self, filename, resolution=128, im_res=64):
        import trimesh

        mc_tri=trimesh.load_mesh(filename)
        v=mc_tri.vertices; f=mc_tri.faces
        mesh2=trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==64 or im_res==128:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh2)
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(v,f)
        mesh2_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==256:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh2_subdiv);
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(mesh2_subdiv.vertices,mesh2_subdiv.faces)
        mesh3_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==256:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh3_subdiv);
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(mesh3_subdiv.vertices,mesh3_subdiv.faces)
        mesh4_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)

        pytorch3d_mesh = self.create_pytorch_mesh(mesh4_subdiv)

        return pytorch3d_mesh

    def project_noise(self, noise, transform, mesh_path=None):
        batch, _, height, width = noise.shape
        assert(batch == 1)  # assuming during inference batch size is 1

        angles = matrix_to_euler_angles(transform[0:1,:,:3], "ZYX")
        azim = float(angles[0][1])
        elev = float(-angles[0][2])

        cameras = create_cameras(azim=azim*180/np.pi,elev=elev*180/np.pi,fov=12.,dist=1)

        renderer = create_depth_mesh_renderer(cameras, image_size=height,
                specular_color=((0,0,0),), ambient_color=((1.,1.,1.),),diffuse_color=((0,0,0),))


        if self.mesh_fn is None or self.mesh_fn != mesh_path:
            self.mesh_fn = mesh_path

        pytorch3d_mesh = self.load_mc_mesh(mesh_path, im_res=height)
        rgb, depth = renderer(pytorch3d_mesh)

        depth_max = depth.max(-1)[0].view(-1) # (NxN)
        depth_valid = depth_max > 0.
        if self.prev_noise is None:
            self.prev_noise = noise
        noise_copy = self.prev_noise.clone()
        noise_copy.view(-1)[depth_valid] = rgb[0,:,:,0].view(-1)[depth_valid]
        noise_copy = noise_copy.reshape(1,1,height,height)  # 1x1xNxN

        return noise_copy


    def forward(self, image, noise=None, transform=None, mesh_path=None):
        batch, _, height, width = image.shape
        if noise is None:
            noise = image.new_empty(batch, 1, height, width).normal_()
        elif self.project:
            noise = self.project_noise(noise, transform, mesh_path=mesh_path)

        return image + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1], project_noise=False):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.noise = NoiseInjection(project=project_noise)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, transform=None, mesh_path=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise, transform=transform, mesh_path=mesh_path)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = upsample
        out_channels = 3
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channels, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False,
                 blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class Decoder(nn.Module):
    def __init__(self, model_opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        # decoder mapping network
        self.size = model_opt.size[1]
        self.style_dim = model_opt.style_dim * 2
        thumb_im_size = model_opt.renderer_spatial_output_dim

        layers = [PixelNorm(),
                   EqualLinear(
                       self.style_dim // 2, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                   )]

        for i in range(4):
            layers.append(
                EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        # decoder network
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * model_opt.channel_multiplier,
            128: 128 * model_opt.channel_multiplier,
            256: 64 * model_opt.channel_multiplier,
            512: 32 * model_opt.channel_multiplier,
            1024: 16 * model_opt.channel_multiplier,
        }

        decoder_in_size = model_opt.renderer_spatial_output_dim[1]

        # image decoder
        self.log_size = int(math.log(self.size, 2))
        self.log_in_size = int(math.log(decoder_in_size, 2))

        self.conv1 = StyledConv(
            model_opt.feature_encoder_in_channels,
            self.channels[decoder_in_size], 3, self.style_dim, blur_kernel=blur_kernel,
            project_noise=model_opt.project_noise)

        self.to_rgb1 = ToRGB(self.channels[decoder_in_size], self.style_dim, upsample=False)

        self.num_layers = (self.log_size - self.log_in_size) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[decoder_in_size]

        for layer_idx in range(self.num_layers + 1):
            res = (0 + 2 * self.log_in_size + 1) // 2
            # res = (layer_idx + 2 * self.log_in_size + 1) // 2
            shape = [1, 1, 2 ** (res), 2 ** (res)]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(self.log_in_size+1, self.log_size+1 + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=False,
                           blur_kernel=blur_kernel, project_noise=model_opt.project_noise)
            )

            self.convs.append(
                StyledConv(out_channel, out_channel, 3, self.style_dim,
                           blur_kernel=blur_kernel, project_noise=model_opt.project_noise)
            )

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

            in_channel = out_channel

        self.n_latent = (self.log_size - self.log_in_size) * 2 + 2 + 2

    def mean_latent(self, renderer_latent):
        latent = self.style(renderer_latent).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def styles_and_noise_forward(self, styles, noise, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False,
                                 randomize_noise=True):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if (truncation < 1):
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[1] + truncation * (style - truncation_latent[1])
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return latent, noise

    def forward(self, features, styles, rgbd_in=None, transform=None,
                return_latents=False, inject_index=None, truncation=1,
                truncation_latent=None, input_is_latent=False, noise=None,
                randomize_noise=True, mesh_path=None):
        latent, noise = self.styles_and_noise_forward(styles, noise, inject_index, truncation,
                                                      truncation_latent, input_is_latent,
                                                      randomize_noise)

        features = torch.sigmoid(features)

        out = self.conv1(features, latent[:, 0], noise=noise[0],
                         transform=transform, mesh_path=mesh_path)

        skip = self.to_rgb1(out, latent[:, 1], skip=rgbd_in)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1,
                           transform=transform, mesh_path=mesh_path)
            out = conv2(out, latent[:, i + 1], noise=noise2,
                                   transform=transform, mesh_path=mesh_path)
            skip = to_rgb(out, latent[:, i + 2], skip=skip)

            i += 2

        out_latent = latent if return_latents else None
        image = skip

        return image, out_latent

############# Volume Renderer Building Blocks & Discriminator ##################
class VolumeRenderDiscConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, activate=False):
        super(VolumeRenderDiscConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias and not activate)

        self.activate = activate
        if self.activate:
            self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
            bias_init_coef = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
            nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)


    def forward(self, input):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: (N,C_out,H_out,W_out)
        :return: Conv2d + activation Result
        """
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_channel = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_y,1)
        yy_channel = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_x,1).transpose(2,3)

        xx_channel = xx_channel / (dim_x - 1)
        yy_channel = yy_channel / (dim_y - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        out = torch.cat([input_tensor, yy_channel, xx_channel], dim=1)

        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(CoordConv2d, self).__init__()

        self.addcoords = AddCoords()
        # self.conv = nn.Conv2d(in_channels, out_channels,
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        # out = input_tensor
        out = self.conv(out)

        return out


class CoordConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True, activate=True):
        super(CoordConvLayer, self).__init__()
        layers = []
        stride = 1
        self.activate = activate
        self.padding = kernel_size // 2 if kernel_size > 2 else 0

        self.conv = CoordConv2d(in_channel, out_channel, kernel_size,
                                padding=self.padding, stride=stride,
                                bias=bias and not activate)

        if activate:
            self.activation = FusedLeakyReLU(out_channel, bias=bias, scale=1)

        bias_init_coef = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
        nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)

    def forward(self, input):
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class VolumeRenderResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = CoordConvLayer(in_channel, out_channel, 3)
        self.conv2 = CoordConvLayer(out_channel, out_channel, 3)
        self.pooling = nn.AvgPool2d(2)
        self.downsample = nn.AvgPool2d(2)
        if out_channel != in_channel:
            self.skip = VolumeRenderDiscConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pooling(out)

        downsample_in = self.downsample(input)
        if self.skip != None:
            skip_in = self.skip(downsample_in)
        else:
            skip_in = downsample_in

        out = (out + skip_in) / math.sqrt(2)

        return out


class VolumeRenderDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        init_size = opt.renderer_spatial_output_dim
        if isinstance(init_size, list):
            if init_size[0] == init_size[1]:
                self.multiplier = 4
            else:
                self.multiplier = 8
            init_size = init_size[1]
        final_out_channel = 1
        channels = {
            2: 512,
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512//2,
            128: 256//2,
            256: 128//2,
            512: 64//2
        }
        self.input_dim = 3
        self.init_size = init_size

        convs = [VolumeRenderDiscConv2d(self.input_dim, channels[self.init_size], 1, activate=True)]

        log_size = int(math.log(self.init_size, 2))

        in_channel = channels[self.init_size]

        for i in range(log_size-1, 0, -1):
            out_channel = channels[2 ** i]

            convs.append(VolumeRenderResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # self.final_conv = VolumeRenderDiscConv2d(in_channel, final_out_channel, 2)
        self.final_conv = torch.nn.Linear(in_channel * self.multiplier, final_out_channel)
        self.in_channel = in_channel


    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out.reshape(-1, self.in_channel * self.multiplier))
        # out = out.mean(-2)
        gan_preds = out[:,0:1]
        gan_preds = gan_preds.view(-1, 1)
        pose_pred = None

        return gan_preds, pose_pred


class VoxelHumanGenerator(nn.Module):
    def __init__(self, model_opt, renderer_opt, blur_kernel=[1, 3, 3, 1], ema=False, full_pipeline=True, voxhuman_name=None):
        super().__init__()
        self.size = model_opt.size
        self.style_dim = model_opt.style_dim
        self.num_layers = 1
        self.train_renderer = not model_opt.freeze_renderer
        self.full_pipeline = full_pipeline
        model_opt.feature_encoder_in_channels = renderer_opt.width
        self.model_opt = model_opt
        self.voxhuman_name = voxhuman_name

        if ema or 'is_test' in model_opt.keys():
            self.is_train = False
        else:
            self.is_train = True

        # volume renderer mapping_network
        layers = []
        for i in range(3):
            layers.append(
                MappingLinear(self.style_dim, self.style_dim, activation="fused_lrelu")
            )

        self.style = nn.Sequential(*layers)

        # volume renderer
        thumb_im_size = model_opt.renderer_spatial_output_dim
        smpl_cfgs = {
            'model_folder': model_opt.smpl_model_folder,
            'model_type': 'smpl',
            'gender': model_opt.smpl_gender,
            'num_betas': 10
        }
        if model_opt.voxhuman_name == 'eva3d_deepfashion':
            VoxHuman_Class = EVA3D_DEEPFASHION_MODEL
        elif model_opt.voxhuman_name == 'eva3d_aist':
            VoxHuman_Class = EVA3D_AIST_MODEL
        else:
            raise NotImplementedError
        self.renderer = VoxHuman_Class(renderer_opt, smpl_cfgs, out_im_res=tuple(model_opt.renderer_spatial_output_dim), style_dim=self.style_dim)

        if self.full_pipeline:
            raise NotImplementedError

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent, device):
        latent_in = torch.randn(n_latent, self.style_dim, device=device)
        renderer_latent = self.style(latent_in)
        renderer_latent_mean = renderer_latent.mean(0, keepdim=True)
        if self.full_pipeline:
            decoder_latent_mean = self.decoder.mean_latent(renderer_latent)
        else:
            decoder_latent_mean = None

        return [renderer_latent_mean, decoder_latent_mean]

    def mean_w_space(self, n_latent, device):
        latent_in = torch.randn(n_latent, self.style_dim, device=device)
        renderer_latent = self.style(latent_in)
        gamma_list = []
        beta_list = []
        for vox in self.renderer.vox_list:
            gamma, beta = vox.network.mapping_network(renderer_latent)
            gamma = gamma.mean(1, keepdim=True)
            beta = beta.mean(1, keepdim=True)
            gamma_list.append(gamma)
            beta_list.append(beta)
        return gamma_list, beta_list

    def project_to_w_space(self, latent_in, truncation, truncation_latent):
        renderer_latent = self.styles_and_noise_forward(latent_in, truncation=truncation, truncation_latent=truncation_latent)
        renderer_latent = torch.cat(renderer_latent, 0)
        gamma_list = []
        beta_list = []
        for vox in self.renderer.vox_list:
            gamma, beta = vox.network.mapping_network(renderer_latent)
            gamma_list.append(gamma)
            beta_list.append(beta)
        return gamma_list, beta_list

    def get_latent(self, input):
        return self.style(input)

    def styles_and_noise_forward(self, styles, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[0] + truncation * (style - truncation_latent[0])
                )

            styles = style_t

        return styles

    def init_forward(self, styles):
        latent = self.styles_and_noise_forward(styles)

        def norm_np_arr(arr):
            return arr / np.linalg.norm(arr)

        def lookat(eye, at, up):
            zaxis = norm_np_arr(eye - at)
            xaxis = norm_np_arr(np.cross(up, zaxis))
            yaxis = np.cross(zaxis, xaxis)
            _viewMatrix = np.array([
                [xaxis[0], yaxis[0], zaxis[0], eye[0]],
                [xaxis[1], yaxis[1], zaxis[1], eye[1]],
                [xaxis[2], yaxis[2], zaxis[2], eye[2]],
                [0       , 0       , 0       , 1     ]
            ])
            return _viewMatrix

        def fix_eye():
            camera_distance = 2
            phi = -np.pi / 2
            theta = np.pi / 2
            return np.array([
                camera_distance * np.sin(theta) * np.cos(phi),
                camera_distance * np.sin(theta) * np.sin(phi),
                camera_distance * np.cos(theta)
            ]), theta, phi, None

        def random_eye_normal():
            camera_distance = np.random.uniform(1.8, 2.5)
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.random.normal(0, np.pi / 4)
            if theta > np.pi / 2 or theta < -np.pi / 2:
                is_front = 0
            else:
                is_front = 1
            return np.array([
                camera_distance * np.sin(theta) * np.cos(phi),
                camera_distance * np.sin(theta) * np.sin(phi),
                camera_distance * np.cos(theta)
            ]), theta, phi, is_front

        def random_eye(is_front=None, distance=None, theta_std=None):
            camera_distance = np.random.uniform(1, 2) if distance is None else distance
            phi = np.random.uniform(0, 2 * np.pi)
            if theta_std == None:
                theta_std = np.pi / 6
            theta = np.random.normal(0, theta_std)
            theta = np.clip(theta, -np.pi / 2, np.pi / 2)
            is_front = np.random.choice(2) if is_front is None else is_front
            if is_front == 0:
                theta += np.pi
            return np.array([
                camera_distance * np.sin(theta) * np.cos(phi),
                camera_distance * np.sin(theta) * np.sin(phi),
                camera_distance * np.cos(theta)
            ]), theta, phi, is_front

        def render_one_batch(v, f, eye, at):
            texture_size = 8
            batch_size = v.shape[0]
            vertices = v.clone()
            faces = torch.from_numpy(f.astype(np.int32)).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            textures = torch.ones(vertices.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
            # rot_mat = torch.from_numpy(np.array(
            #     [[ 1.,  0.,  0.],
            #     [ 0.,  0., -1.],
            #     [ 0.,  1.,  0.]], dtype=np.float32)).cuda()
            # vertices = torch.matmul(vertices, rot_mat)
            renderer = nr.Renderer(camera_mode='look').cuda()
            renderer.eye = eye.float()
            renderer.camera_direction = (at - eye) / torch.norm(at - eye)
            renderer.light_intensity_directional = 0.0
            renderer.light_intensity_ambient = 1.0
            images, _, _ = renderer(vertices, faces, textures)
            detached_images = images.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            # detached_images = detached_images[:, ::-1]
            return detached_images.copy()

        # setup camera
        eye, theta, phi, _ = random_eye_normal()
        # eye, theta, phi, _ = fix_eye()
        at = np.zeros([3]).astype(np.float32)
        eye = eye.astype(np.float32)
        eye += at
        cam_poses = lookat(eye, at, np.array([0, 0, 1])).reshape(1, 4, 4)
        focals = np.array([0.5 * 256 / np.tan(0.5 * 1.0471975511965976)])
        cam_poses = torch.from_numpy(cam_poses).float().cuda()
        focals = torch.from_numpy(focals).float().cuda()

        #setup smpl model
        beta = torch.zeros([1, 10]).cuda()
        trans = torch.zeros([1, 3]).cuda()
        theta = torch.zeros([24, 3]).cuda().reshape(-1, 3)
        theta[0 ,0] = np.pi / 2
        theta[17, 2] = np.pi / 3
        theta[16, 2] = -np.pi / 3
        pose_rot = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        so = self.renderer.smpl_model(betas = beta, body_pose = pose_rot[:, 1:], global_orient = pose_rot[:, 0, :, :].view(1, 1, 3, 3))
        v = so['vertices'].clone()
        f = self.renderer.smpl_model.faces.copy()
        del so
        true_rgb = torch.from_numpy(render_one_batch(v, f, torch.from_numpy(eye).cuda(), torch.from_numpy(at).cuda()))
        true_rgb = F.interpolate(true_rgb.reshape(256, 256, 3).permute(2, 0, 1).unsqueeze(0), \
                                size=(256, 256)).permute(0, 2, 3, 1).cuda()
        true_rgb = true_rgb[:, :, 64:192, :]

        rgb_map, feature_map, _, _, _, _ = self.renderer(cam_poses, focals, beta, theta.unsqueeze(0), trans, styles=latent[0], no_white_bg=True)

        rgb_map = (rgb_map + 1) / 2
        
        imageio.imwrite('test/true_rgb.jpg', true_rgb.cpu().numpy().reshape(256, 128, 3))
        imageio.imwrite('test/rgb_map.jpg', rgb_map.detach().cpu().numpy().reshape(256, 128, 3))
        # imageio.imwrite('test/visibility.jpg', feature_map.detach().cpu().numpy().reshape(128, 64))
        # import pdb; pdb.set_trace()


        return rgb_map, true_rgb

    def forward(self, styles, cam_poses, focals, beta, theta, trans,
                inject_index=None, truncation=1, truncation_latent=None,
                input_is_latent=False, return_sdf=False, return_xyz=False, return_eikonal=False,
                return_normal=False, return_mask=False, inv_Ks=None, return_sdf_xyz=False,
                w_space=False, gamma_list=None, beta_list=None, fix_viewdir=False):
        if w_space:
            latent = [None, None]
        else:
            latent = self.styles_and_noise_forward(styles, inject_index, truncation,
                                                    truncation_latent, input_is_latent)

        if inv_Ks is None:
            if w_space:
                thumb_rgb, features, _sdf, mask, xyz, eikonal_term = self.renderer(cam_poses, focals, beta, theta, trans, styles=latent[0], \
                                                                                return_eikonal=return_eikonal,
                                                                                w_space=w_space, gamma_list=gamma_list, beta_list=beta_list,
                                                                                fix_viewdir=fix_viewdir)
            else:
                thumb_rgb, features, _sdf, mask, xyz, eikonal_term = self.renderer(cam_poses, focals, beta, theta, trans, styles=latent[0], \
                                                                                return_eikonal=return_eikonal, fix_viewdir=fix_viewdir)
        else:
            thumb_rgb, features, _sdf, mask, xyz, eikonal_term = self.renderer(cam_poses, focals, beta, theta, trans, styles=latent[0], \
                                                                            return_eikonal=return_eikonal, inv_Ks=inv_Ks,
                                                                            w_space=w_space, gamma_list=gamma_list, beta_list=beta_list,
                                                                            fix_viewdir=fix_viewdir)

        if return_sdf_xyz:
            sdf = _sdf[0]
            sdf_xyz = _sdf[1]
        else:
            sdf = _sdf[0]

        if self.full_pipeline:
            raise NotImplementedError
        else:
            rgb = None

        out = (rgb, thumb_rgb)
        if return_xyz:
            out += (xyz,)
        if return_sdf:
            out += (sdf,)
        if return_eikonal:
            out += (eikonal_term,)
        if return_mask:
            out += (mask,)
        if return_normal:
            out += (features, )
        if return_sdf_xyz:
            out += (sdf_xyz, )

        return out
