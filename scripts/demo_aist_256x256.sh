#!/bin/bash
set -x

python generation_demo.py --batch 1 --chunk 1 --expname 256x256_aist --dataset_path demodataset --depth 5 --width 128 --style_dim 128 --renderer_spatial_output_dim 512 256 --input_ch_views 3 --white_bg --voxhuman_name eva3d_deepfashion --deltasdf --N_samples 28 --ckpt 340000 --identities 5 --truncation_ratio 0.5 --is_aist #--render_video

