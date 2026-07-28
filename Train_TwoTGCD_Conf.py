import os
import numpy as np
import torchvision
import torch
from torch.optim import Adam
import env.Config as config
from utils.TwoDenoiseModel import *
from utils.Diffusion import *
from TrainNetworkHelper_TwoTG import *
import matplotlib
import random
from env.Dataset import *

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization
    torch.backends.cudnn.deterministic = True  # Ensure deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    setup_seed(0)
    denoise_model = SPDenoiser(
        dim=64,
        cond_dim=512*4*4,
        group1_size=config.group1_size,
        group1_dim=config.group1_dim,
        group2_size=config.group2_size,
        group2_dim=config.group2_dim,
        time_emb_dim=256,
        hidden_dim=512
    ).to(config.device)
    diffusion = SPDiff(
        denoise_model,
        timesteps=config.timesteps,  # number of steps
        loss_type='l1'
    ).to(config.device)

    train_dataset = TCDataset_HisPR_TwoTG(config.train_k8_path)
    valid_dataset = TCDataset_HisPR_TwoTG(config.valid_k8_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )

    optimizer = Adam(diffusion.parameters(), lr=5e-5)

    Trainer = CondDiffusionTrainer_GroupGrad(epoches=config.epochs,
                                     train_loader=train_dataloader,
                                     valid_loader=valid_dataloader,
                                     optimizer=optimizer,
                                     device=config.device,
                                     timesteps=config.timesteps)

    CDModel = Trainer.forward(diffusion, model_save_path=config.model_output_dir, save_fig_dir=config.model_output_dir)


