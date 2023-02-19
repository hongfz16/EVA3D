#!/bin/bash
set -x

python generation_demo.py --batch 1 --chunk 1 --expname 512x256_ubcfashion --dataset_path demodataset --depth 5 --width 128 --style_dim 128 --renderer_spatial_output_dim 512 256 --input_ch_views 3 --white_bg --voxhuman_name eva3d_deepfashion --deltasdf --N_samples 28 --ckpt 620000 --identities 5 --truncation_ratio 0.5 #--render_video

