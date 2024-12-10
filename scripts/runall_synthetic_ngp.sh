# Hard point mining
CUDA_VISIBLE_DEVICES=0 python main.py data/nerf_synthetic/lego/ --workspace trial_lego        --hsm --wandb --adaptive_num_rays --ckpt scratch --enable_cam_center --enable_cam_near_far --min_near 0.2 --lambda_tv 0 --lambda_distort 0 --eval_cnt 1 --save_cnt 1 --downscale 4 -O --background random --bound 8
