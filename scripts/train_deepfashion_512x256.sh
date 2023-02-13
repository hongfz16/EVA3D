MASTER_PORT=$((12000 + $RANDOM % 20000))
NUM_GPU=8
set -x

python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} --master_port=${MASTER_PORT} train_deepfashion.py --batch 1 --chunk 1 --expname train_deepfashion_512x256 --dataset_path datasets/DeepFashion --depth 5 --width 128 --style_dim 128 --renderer_spatial_output_dim 512 256 --input_ch_views 3 --white_bg --r1 300 --voxhuman_name eva3d_deepfashion --random_flip --eikonal_lambda 0.5 --small_aug --iter 1000000 --adjust_gamma --gamma_lb 20 --min_surf_lambda 1.5 --deltasdf --gaussian_weighted_sampler --sampler_std 15 --N_samples 28
