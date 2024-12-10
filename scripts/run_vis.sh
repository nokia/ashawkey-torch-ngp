# baseline
CUDA_VISIBLE_DEVICES=0 python main.py data/bonsai/ --workspace trial_ngp_360_bonsai_test_video --adaptive_num_rays --ckpt scratch --enable_cam_center --enable_cam_near_far --min_near 0.2 --lambda_tv 0 --lambda_distort 0 --eval_cnt -1 --save_cnt 1 --downscale 2 -O --background random --bound 8 --val_checkpoint 5 --max_training_time 4 --seed 1

# hard mining
CUDA_VISIBLE_DEVICES=0 python main.py data/bonsai/ --workspace trial_ngp_360_bonsai_hsm_test_video  --hsm --adaptive_num_rays --ckpt scratch --enable_cam_center --enable_cam_near_far --min_near 0.2 --lambda_tv 0 --lambda_distort 0 --eval_cnt -1 --save_cnt 1 --downscale 2 -O --background random --bound 8 --val_checkpoint 5 --max_training_time 4 --seed 1
