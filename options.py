import configargparse
from munch import *
import numpy as np
from pdb import set_trace as st

class BaseOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./datasets/DeepFashion')
        dataset.add_argument("--random_flip", action='store_true')
        dataset.add_argument("--gaussian_weighted_sampler", action='store_true')
        dataset.add_argument("--sampler_std", type=float, default=15)

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        experiment.add_argument("--expname", type=str, default='debug', help='experiment name')
        experiment.add_argument("--ckpt", type=str, default='300000', help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")

        # Training loop options
        training = self.parser.add_argument_group('training')
        training.add_argument("--checkpoints_dir", type=str, default='./checkpoint', help='checkpoints directory name')
        training.add_argument("--iter", type=int, default=300000, help="total number of training iterations")
        training.add_argument("--batch", type=int, default=4, help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=4, help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--val_n_sample", type=int, default=8, help="number of test samples generated during training")
        training.add_argument("--d_reg_every", type=int, default=16, help="interval for applying r1 regularization to the StyleGAN generator")
        training.add_argument("--g_reg_every", type=int, default=4, help="interval for applying path length regularization to the StyleGAN generator")
        training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        training.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        training.add_argument("--lr", type=float, default=0.002, help="learning rate")
        training.add_argument("--r1", type=float, default=300, help="weight of the r1 regularization")
        training.add_argument("--eikonal_lambda", type=float, default=0.5, help="weight of the eikonal regularization")
        training.add_argument("--min_surf_lambda", type=float, default=1.5, help="weight of the minimal surface regularization")
        training.add_argument("--min_surf_beta", type=float, default=100.0, help="weight of the minimal surface regularization")
        training.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        training.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        training.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        training.add_argument("--small_aug", action='store_true')
        training.add_argument("--adjust_gamma", action='store_true', default=False)
        training.add_argument("--gamma_lb", type=float, default=20)
        training.add_argument("--glr", type=float, default=2e-5)
        training.add_argument("--dlr", type=float, default=2e-4)
        training.add_argument("--deltasdf", action='store_true', default=False)
        training.add_argument("--fid_path", type=str, default='')

        # Inference Options
        inference = self.parser.add_argument_group('inference')
        inference.add_argument("--results_dir", type=str, default='./evaluations', help='results/evaluations directory name')
        inference.add_argument("--truncation_ratio", type=float, default=0.5, help="truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results")
        inference.add_argument("--truncation_mean", type=int, default=10000, help="number of vectors to calculate mean for the truncation")
        inference.add_argument("--identities", type=int, default=16, help="number of identities to be generated")
        inference.add_argument("--num_views_per_id", type=int, default=1, help="number of viewpoints generated per identity")
        inference.add_argument("--no_surface_renderings", action="store_true", help="when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video")
        inference.add_argument("--fixed_camera_angles", action="store_true", help="when true, the generator will render indentities from a fixed set of camera angles.")
        inference.add_argument("--azim_video", action="store_true", help="when true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory.")

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--size", type=int, nargs="+", default=[256, 128], help="image sizes for the model")
        model.add_argument("--style_dim", type=int, default=256, help="number of style input dimensions")
        model.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the StyleGAN decoder. config-f = 2, else = 1")
        model.add_argument("--n_mlp", type=int, default=8, help="number of mlp layers in stylegan's mapping network")
        model.add_argument("--lr_mapping", type=float, default=0.01, help='learning rate reduction for mapping network MLP layers')
        model.add_argument("--renderer_spatial_output_dim", type=int, nargs="+", default=[128, 64], help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument("--project_noise", action='store_true', help='when true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). warning: processing time significantly increases with this flag to ~20 minutes per video.')
        model.add_argument("--smpl_model_folder", type=str, default="smpl_models", help='path to smpl model folder')
        model.add_argument("--smpl_gender", type=str, default="neutral")
        model.add_argument("--voxhuman_name", type=str, default=None)

        # Camera options
        camera = self.parser.add_argument_group('camera')
        camera.add_argument("--uniform", action="store_true", help="when true, the camera position is sampled from uniform distribution. Gaussian distribution is the default")
        camera.add_argument("--azim", type=float, default=0.3, help="camera azimuth angle std/range in Radians")
        camera.add_argument("--elev", type=float, default=0.15, help="camera elevation angle std/range in Radians")
        camera.add_argument("--fov", type=float, default=6, help="camera field of view half angle in Degrees")
        camera.add_argument("--dist_radius", type=float, default=0.12, help="radius of points sampling distance from the origin. determines the near and far fields")

        # Volume Renderer options
        rendering = self.parser.add_argument_group('rendering')
        # MLP model parameters
        rendering.add_argument("--depth", type=int, default=8, help='layers in network')
        rendering.add_argument("--width", type=int, default=256, help='channels per layer')
        # Volume representation options
        rendering.add_argument("--no_sdf", action='store_true', help='By default, the raw MLP outputs represent an underline signed distance field (SDF). When true, the MLP outputs represent the traditional NeRF density field.')
        rendering.add_argument("--no_z_normalize", action='store_true', help='By default, the model normalizes input coordinates such that the z coordinate is in [-1,1]. When true that feature is disabled.')
        rendering.add_argument("--static_viewdirs", action='store_true', help='when true, use static viewing direction input to the MLP')
        rendering.add_argument("--is_aist", action='store_true')
        # Ray intergration options
        rendering.add_argument("--N_samples", type=int, default=24, help='number of samples per ray')
        rendering.add_argument("--no_offset_sampling", action='store_true', help='when true, use random stratified sampling when rendering the volume, otherwise offset sampling is used. (See Equation (3) in Sec. 3.2 of the paper)')
        rendering.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        rendering.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
        rendering.add_argument("--force_background", action='store_true', help='force the last depth sample to act as background in case of a transparent ray')
        # Set volume renderer outputs
        rendering.add_argument("--return_xyz", action='store_true', help='when true, the volume renderer also returns the xyz point could of the surface. This point cloud is used to produce depth map renderings')
        rendering.add_argument("--return_sdf", action='store_true', help='when true, the volume renderer also returns the SDF network outputs for each location in the volume')
        rendering.add_argument("--stepsize", type=float, default=0.02, help='rendering step size')
        rendering.add_argument("--white_bg", action='store_true')
        rendering.add_argument("--input_ch_views", type=int, default=3)
        # inference options
        rendering.add_argument("--render_video", action='store_true')

        self.initialized = True

    def parse(self):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()
        try:
            args = self.parser.parse_args()
        except: # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt
