#© 2024 Nokia
#Licensed under the BSD 3 Clause license
#SPDX-License-Identifier: BSD-3-Clause

import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
import wandb

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

try:
    from torchmetrics.functional import structural_similarity_index_measure
except: # old versions
    from torchmetrics.functional import ssim as structural_similarity_index_measure

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):

    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    # construct camera poses by lookat
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=-1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=-1))

    ### construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
    poses[:, :3, 3] = vertices

    return poses


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    device = poses.device
    
    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:
       
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten


        else: # random sampling
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()

    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i) # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(1) # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d) # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def visualize_rays(rays_o, rays_d):
    
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'
    
    def name(self):
        return "PSNR"

class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

        return v
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'
    
    def name(self):
        return "LPIPS"

class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'
    
    def name(self):
        return "SSIM"

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 save_interval=1, # save once every $ epoch (independently from eval)
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.opt = opt
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        # sampling parameters
        self.tau = 1.0
        self.val_timer = time.time()
        self.accumulated_training_time = 0 # used to monitor training time (excluding evaluation time)
        self.gt_saved = 0
        self.max_memory_usage = 0 # used to create a memory monitoring when visualizing the training
        self.cumulative_iteration_time = 0
        self.n_iters = 0

        # try out torch 2.0
        if torch.__version__[0] == '2':
            model = torch.compile(model)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.log(opt)
        self.log(self.model)

        if self.workspace is not None:

            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        self.opt.constant_memory_load = torch.cuda.memory_allocated()
        self.opt.running_max_mem = 0
        self.log(f"Constant Memory Usage (Bytes) = {self.opt.constant_memory_load}")


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        index = data['index'] # [1/N]
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        images = data['images'] # [N, 3/4]

        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device) # [N, 3], pixel-wise random.
        else: # white / last_sample
            bg_color = 1

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0
        
        outputs = self.model.render(rays_o, rays_d, gt_image=gt_rgb, index=index, bg_color=bg_color, perturb=True, cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal, step=self.global_step)

        # MSE loss
        pred_rgb = outputs['image']
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [N, 3] --> [N]
        loss = loss.mean()

        if self.opt.wandb:
            wandb.log({"rgb loss" : loss}, step=self.global_step)


        # proposal loss to update the samplers in nerfacto
        if 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
            if self.opt.wandb:
                wandb.log({"proposal loss" : self.opt.lambda_proposal * outputs['proposal_loss'].mean()}, step=self.global_step)
            loss = loss + self.opt.lambda_proposal * outputs['proposal_loss']

        # extra loss
        if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
            loss = loss + self.opt.lambda_distort * outputs['distort_loss']

        if self.opt.lambda_entropy > 0:
            w = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
            loss = loss + self.opt.lambda_entropy * (entropy.mean())

        # adaptive num_rays
        if self.opt.adaptive_num_rays:
            self.opt.num_rays = int(round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))
            
        return pred_rgb, gt_rgb, loss
    

    def hsm_train_step(self, data):
        
        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        index = data['index'] # [1/N]
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        images = data['images'] # [N, 3/4]

        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device) # [N, 3], pixel-wise random.
        else: # white / last_sample
            bg_color = 1

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        if shading != 'full':
            # Throw a warning only once
            warnings.warn("Shading is not to 'full' is not tested with hard sample mining.", UserWarning)
        if self.opt.O:
            output, output_gradient = self.model.run_hsm(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True, cam_near_far=cam_near_far, shading=shading, gt_image=gt_rgb, global_step=self.global_step)
        if self.opt.O2:
            update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0
            output, output_gradient, prop_loss = self.model.run_hsm2(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True, cam_near_far=cam_near_far, shading=shading, gt_image=gt_rgb, update_proposal=update_proposal, global_step=self.global_step)
            if update_proposal:
                if self.opt.wandb:
                    wandb.log({"proposal loss" : prop_loss}, step=self.global_step)
                self.scaler.scale(prop_loss).backward() # proposal samplers form a separate graph
        
        self.scaler.scale(output).backward(gradient=output_gradient)
        self.post_train_step() # for TV loss...
        self.scaler.step(self.optimizer)
        self.scaler.update()


    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            self.model.apply_total_variation(self.opt.lambda_tv)
        
        if self.opt.lambda_wd > 0:
            self.model.apply_weight_decay(self.opt.lambda_wd)
                

    def eval_step(self, data):

        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        images = data['images'] # [H, W, 3/4]
        index = data['index'] # [1/N]
        H, W, C = images.shape

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        # eval with fixed white background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=False, cam_near_far=cam_near_far)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, shading='full'):  

        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        index = data['index'] # [1/N]
        H, W = data['H'], data['W']

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=perturb, cam_near_far=cam_near_far, shading=shading)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)

        return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=128, decimate_target=1e5, dataset=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=resolution, decimate_target=decimate_target, dataset=dataset)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        start_t = time.time()
        self.val_timer = time.time() # start validation timer
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            ret_code = self.train_one_epoch(train_loader, valid_loader=valid_loader)
            if ret_code==0:
                print(f"Accumulated training time: {self.accumulated_training_time} seconds")
                self.save_checkpoint(full=True, best=False)
                break

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                preds, preds_depth = self.test_step(data)
                pred = preds.detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0) # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0) # [N, H, W]

            # fix ffmpeg not divisible by 2
            all_preds = np.pad(all_preds, ((0, 0), (0, 1 if all_preds.shape[1] % 2 != 0 else 0), (0, 1 if all_preds.shape[2] % 2 != 0 else 0), (0, 0)))
            all_preds_depth = np.pad(all_preds_depth, ((0, 0), (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0)))

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=24, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=24, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0 and self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()            
            
            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net = self.train_step(data)
            
            loss = loss_net
         
            self.scaler.scale(loss).backward()

            self.post_train_step() # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss_net.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, shading='full'):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'mvp': mvp,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'index': [0],
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            # here spp is used as perturb random seed! (but not perturb the first sample)
            preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, shading=shading)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.unsqueeze(0).permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).squeeze(0).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(0).unsqueeze(1), size=(H, W), mode='nearest').squeeze(0).squeeze(1)

        pred = preds.detach().cpu().numpy()
        pred_depth = preds_depth.detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs
    
    def get_b_sample(self, ranking_metric, b):
        probs = ranking_metric
        return torch.multinomial(probs, b, replacement=False)


    def mask_data(self, data, mask_indices):

        for key in data.keys():
            if (not isinstance(data[key], int)) and len(data[key]) > 1:
                data[key] = data[key][mask_indices]


        return data

    def non_maximum_supression(self, data, loss):
        # data["i"] is of size N
        box_side_in_pixels = 10
        boxes = torch.cat([(data["i"].unsqueeze(-1)-box_side_in_pixels).clamp(min=0),
                            (data["j"].unsqueeze(-1)-box_side_in_pixels).clamp(min=0),
                              data["i"].unsqueeze(-1)+box_side_in_pixels,
                                data["j"].unsqueeze(-1)+box_side_in_pixels], dim=1).float() + (data["index"] * max(data["H"], data["W"])).unsqueeze(-1) # torch.norm(data["rays_o"], dim=1, keepdim=True)*100 # should be [N,4]
        return torchvision.ops.batched_nms(boxes, loss, data['index'].int(), 0.7)
        # return torchvision.ops.nms(boxes, loss, 0.8) # non-batch version is faster


    def eval_timestep(self, valid_loader):
        # checkpoint monitor
        if ((time.time() - self.val_timer) >  self.opt.val_checkpoint) and self.opt.eval_cnt != -1:
            training_time = (time.time() - self.val_timer)
            self.accumulated_training_time += training_time
            self.log(f"Average Training Speed (Iterations per Second) = {self.n_iters / training_time}")
            self.log(f"Running Max Memory Usage (MB) = {self.opt.running_max_mem}")
            if self.opt.wandb:
                wandb.log({"training time" : self.accumulated_training_time, "Running Max Memory Usage (MB)" : self.opt.running_max_mem}, step=self.global_step)
            self.model.eval()
            self.evaluate_one_epoch(valid_loader)
            self.save_checkpoint(full=True, best=False)
            self.n_iters = 0 # reset iteration counter
            self.opt.running_max_mem = 0 # reset memory usage
            self.model.train()
            self.val_timer = time.time() # reset validation timer
            torch.cuda.reset_peak_memory_stats() # do not monitor eval memory

    
    
    
    def train_one_epoch(self, loader, valid_loader=None):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1
            self.n_iters += 1


            self.optimizer.zero_grad()    

            # forward pass
            if self.opt.hsm:
                self.hsm_train_step(data)

                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()
                
                if self.opt.wandb:
                    wandb.log({"Max memory used per iteration (MB)" : (torch.cuda.max_memory_allocated()-self.opt.constant_memory_load)/1e6}, step=self.global_step)
                
                pbar.set_description(f"Memory Used={(torch.cuda.max_memory_allocated()-self.opt.constant_memory_load)/1e6:.0f} MB")
                self.opt.running_max_mem = max(torch.cuda.max_memory_allocated(), self.opt.running_max_mem)
                pbar.update(loader.batch_size)
                
                self.eval_timestep(valid_loader=valid_loader)
                torch.cuda.reset_peak_memory_stats()
                if (self.accumulated_training_time / 60) >= self.opt.max_training_time:
                    return 0
                continue
            
            preds, truths, loss = self.train_step(data)
            # backward pass
            self.scaler.scale(loss.mean()).backward()
            self.post_train_step() # for TV loss...
            self.scaler.step(self.optimizer)
            self.scaler.update()




            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.mean().item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                pbar.update(loader.batch_size)

            pbar.set_description(f"Memory Used={(torch.cuda.max_memory_allocated()-self.opt.constant_memory_load)/1e6:.0f} MB")
            if self.opt.wandb:
                wandb.log({"Max memory used per iteration (MB)" : (torch.cuda.max_memory_allocated()-self.opt.constant_memory_load)/1e6}, step=self.global_step)
            
            self.opt.running_max_mem = max(torch.cuda.max_memory_allocated(), self.opt.running_max_mem)
            self.eval_timestep(valid_loader=valid_loader)
            torch.cuda.reset_peak_memory_stats()
            if (self.accumulated_training_time / 60) >= self.opt.max_training_time:
                return 0


        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")
        return 1

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                if self.opt.val_img!=-1 and self.local_step!=self.opt.val_img:
                    continue

                preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    
                    metric_vals = []
                    for metric in self.metrics:
                        metric_val = metric.update(preds, truths)
                        metric_vals.append(metric_val)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_{int(self.accumulated_training_time):04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_{int(self.accumulated_training_time):04d}_depth.png')
                    save_path_error = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_{int(self.accumulated_training_time):04d}_error_{metric_vals[0]:.2f}.png') # metric_vals[0] should be the PSNR

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth.detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    truth = truths.detach().cpu().numpy()
                    truth = (truth * 255).astype(np.uint8)
                    error = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(-1).astype(np.uint8)
                    
                    if not self.gt_saved:
                        # truth = (truth * 255).astype(np.uint8)
                        gt_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.local_step:04d}_gt_rgb.png')
                        cv2.imwrite(gt_path, cv2.cvtColor(truth, cv2.COLOR_RGB2BGR))



                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)
                    cv2.imwrite(save_path_error, error)

                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        self.gt_saved = 1 # save gt only once in the beginning

        if self.local_rank == 0:
            pbar.close()
            
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.opt.wandb:
                    wandb.log({f"{metric.name()}" : metric.measure(), "training time" : int(self.accumulated_training_time)}, step=self.global_step)
                    wandb.log({f"{metric.name()} per iteration" : metric.measure()}, step=self.global_step)
                    wandb.log({"Max Memory Usage (Bytes)": self.max_memory_usage, "training time" : int(self.accumulated_training_time)}, step=self.global_step)
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        return average_loss

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    # if 'density_grid' in state['model']:
                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):

        if checkpoint is None: # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth')) 

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
    

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")