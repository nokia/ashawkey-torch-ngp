# llff
# python main.py data/nerf_llff_data/fern --workspace trial_llff_fern -O --data_format colmap --bound 4 --visibility_mask_dilation 50 --downscale 4 --max_training_time 8 --wandb --ckpt scratch
# python main.py data/nerf_llff_data/fern --workspace trial_llff_fern_hsm -O --data_format colmap --bound 4 --visibility_mask_dilation 50 --downscale 4 --hsm --max_training_time 8 --wandb --ckpt scratch

# synthetic
# python main.py data/nerf_synthetic/lego/ --workspace trial_syn_lego_hsm -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --hsm --data_format nerf --wandb --ckpt scratch --background white
# python main.py data/nerf_synthetic/lego/ --workspace trial_syn_lego -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --data_format nerf --wandb --ckpt scratch --background white

# python main.py data/nerf_synthetic/mic/ --workspace trial_syn_mic_hsm/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --hsm --data_format nerf --ckpt scratch --background white
# python main.py data/nerf_synthetic/mic/ --workspace trial_syn_mic/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --data_format nerf --ckpt scratch --background white

# python main.py data/nerf_synthetic/materials/ --workspace trial_syn_materials_hsm/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --hsm --data_format nerf --ckpt scratch --background white
# python main.py data/nerf_synthetic/materials/ --workspace trial_syn_materials/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --data_format nerf --ckpt scratch --background white

# python main.py data/nerf_synthetic/chair/ --workspace trial_syn_chair_hsm/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --hsm --data_format nerf --ckpt scratch --background white
# python main.py data/nerf_synthetic/chair/ --workspace trial_syn_chair/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --data_format nerf --ckpt scratch --background white

# python main.py data/nerf_synthetic/hotdog/ --workspace trial_syn_hotdog_hsm/ -O --bound 1 --scale 0.7 --dt_gamma 0 --wandb --hsm --data_format nerf  --ckpt scratch --background white
# python main.py data/nerf_synthetic/hotdog/ --workspace trial_syn_hotdog/ -O --bound 1 --scale 0.7 --dt_gamma 0 --wandb --data_format nerf  --ckpt scratch --background white

# python main.py data/nerf_synthetic/ficus/ --workspace trial_syn_ficus_hsm/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --hsm --data_format nerf --ckpt scratch --background white
# python main.py data/nerf_synthetic/ficus/ --workspace trial_syn_ficus/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --data_format nerf --ckpt scratch --background white

python main.py data/nerf_synthetic/drums/ --workspace trial_syn_drums_hsm/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --hsm --data_format nerf --ckpt scratch --background white
python main.py data/nerf_synthetic/drums/ --workspace trial_syn_drums/ -O --bound 1 --scale 0.8 --dt_gamma 0 --wandb --data_format nerf --ckpt scratch --background white

python main.py data/nerf_synthetic/ship/ --workspace trial_syn_ship_hsm/ -O --bound 1 --scale 0.7 --dt_gamma 0 --wandb --hsm --data_format nerf --ckpt scratch --background white
python main.py data/nerf_synthetic/ship/ --workspace trial_syn_ship/ -O --bound 1 --scale 0.7 --dt_gamma 0 --wandb --data_format nerf --ckpt scratch --background white