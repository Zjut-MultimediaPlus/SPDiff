import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import math
from functools import partial
import env.Config as config
from utils.Condition_Encoder import *   # utils.
from typing import Tuple, Dict, Literal

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResCepx2CoefMultiDenoiser(nn.Module):
    def __init__(self, group_size, group_dim, cond_dim=8192, time_emb_dim=128, hidden_dim=512):
        super().__init__()
        self.group_dim = group_dim
        self.group_size = group_size
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.cep2gweight = nn.Linear(group_dim, 1)
        self.FexpGdim = nn.Linear(hidden_dim + time_emb_dim, hidden_dim + time_emb_dim)
        self.film_net = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, group_dim * 2),
            nn.GELU()
        )

        # Cross-Attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.attn_proj_q = nn.Linear(group_dim, hidden_dim)
        self.attn_proj_kv = nn.Linear(hidden_dim + time_emb_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, group_dim)

        # concat MLP
        self.concat_mlp = nn.Sequential(
            nn.Linear(group_dim + hidden_dim + time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, group_dim)
        )

    def forward(self, x, c, t_emb, xcepx):
        """
        x: (b, group_size, group_dim)
        cond: (b, F)
        t: (b, time_emb_dim)
        """
        b = x.shape[0]

        c = c.unsqueeze(dim=1)  # (b, 1, F)
        t_emb = t_emb.unsqueeze(dim=1)  # (b, 1, time_emb_dim)
        ct_emb = torch.cat([c, t_emb], dim=-1)       # (b, 1, F+time_emb_dim)
        gw = self.cep2gweight(xcepx)     # (b, g, 1)
        g_ct_emb = torch.einsum("bgo,bof->bgf", gw, ct_emb)     # (b, g, F+time_emb_dim)
        g_ct_emb = self.FexpGdim(g_ct_emb)
        # --------- FiLM ---------
        gamma_beta = self.film_net(g_ct_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)       # (b, g, group_dim)
        x1 = gamma * x + beta       # (b, group_size, group_dim)

        # --------- Cross-Attention ---------
        x_q = self.attn_proj_q(x)  # (b, group_size, hidden_dim)
        kv = self.attn_proj_kv(g_ct_emb)         # (b,group_size,hidden_dim)
        x2, _ = self.attn(x_q, kv, kv)        # (b,group_size,hidden_dim)
        x2 = self.final_proj(x2)        # (b,group_size,hidden_dim)

        # --------- concat MLP ---------
        x3 = self.concat_mlp(torch.cat([x, g_ct_emb], dim=-1))

        denoise_x = x1 + x2 + x3      # b, g, d

        return denoise_x

class SPDenoiser(nn.Module):
    def __init__(
            self,
            dim=64,
            cond_dim=512 * 4 * 4,
            group1_size=config.group1_size,
            group1_dim=config.group1_dim,
            group2_size=config.group2_size,
            group2_dim=config.group2_dim,
            time_emb_dim=None,
            hidden_dim=512,
    ):
        super().__init__()

        self.group1_size = group1_size
        self.group1_dim = group1_dim
        self.group2_size = group2_size
        self.group2_dim = group2_dim
        self.has_cond = True
        self.flatten = nn.Flatten()

        tdim = default(time_emb_dim, dim * 4)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, tdim),
            nn.GELU(),
            nn.Linear(tdim, tdim)
        )

        self.cond_enc = MultiCondEnc(ratio_num=5)
        self.cond_dim = int(cond_dim)

        self.cond_proj1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_proj2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.null_cond_emb = nn.Parameter(torch.randn(1, self.cond_dim))
        self.group1_denoiser = ResCepx2CoefMultiDenoiser(group_size=group1_size, group_dim=group1_dim, cond_dim=cond_dim,
                                                   time_emb_dim=tdim, hidden_dim=hidden_dim)
        self.group2_denoiser = ResCepx2CoefMultiDenoiser(group_size=group2_size, group_dim=group2_dim, cond_dim=cond_dim,
                                                   time_emb_dim=tdim, hidden_dim=hidden_dim)

    def CondTemb(
            self,
            time,
            cond_visual=None, cond_ratio=None,
            null_cond_prob=0.,
    ):
        batch = time.shape[0]
        device = time.device

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if exists(t) and t.dim() == 1:
            t = t.unsqueeze(0)

        mask = prob_mask_like((batch,), null_cond_prob, device=device)
        if exists(self.cond_enc) and exists(cond_visual):
            vcond = self.cond_enc(cond_visual, cond_ratio)  # (b, cond_dim)
            vcond = self.flatten(vcond)  # (b, cond_dim)
        else:
            vcond = torch.zeros((batch, self.cond_dim), device=device)

        cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb.expand(batch, -1), vcond)

        t_emb = t if exists(t) else torch.zeros((batch, self.time_mlp[-1].out_features), device=device)

        c1 = self.cond_proj1(cond)  # (b, F)
        c2 = self.cond_proj2(cond)  # (b, F)
        return c1, c2, t_emb

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        res_denoise_x1, res_denoise_x2 = self.forward(*args)
        return res_denoise_x1, res_denoise_x2

    def forward(
            self, x1, x2,
            c1, c2, t_emb, cepx1, cepx2
    ):
        # (b, g, d)
        denoise_x1 = self.group1_denoiser(x1, c1, t_emb, cepx1)
        denoise_x2 = self.group2_denoiser(x2, c2, t_emb, cepx2)

        return denoise_x1, denoise_x2