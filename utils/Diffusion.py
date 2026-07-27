import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import env.Config as config

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

def extract(a, t, x_shape):
    b, *_ = t.shape
    a = a.to(config.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)
class StableSoftmax(nn.Module):
    def __init__(self, dim: int = -1, mix_uniform_eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.mix = mix_uniform_eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        dtype = logits.dtype
        x = logits.float()
        x = x - x.max(dim=self.dim, keepdim=True).values
        p = torch.softmax(x, dim=self.dim)  # float32

        K = p.size(self.dim)
        if self.mix > 0:
            p = (1.0 - self.mix) * p + self.mix * (1.0 / K)
        return p.to(dtype)


class MatchConcept(nn.Module):
    def __init__(
        self,
        vfdim: int,
        cfdim: int,
        group_size: int,
        group_dim: int,
        tau: float = 0.07,
        lambda_orth: float = 1.0,
        activation: str = "softmax",
        st_topk: bool = True,
        clamp_negative: bool = False,
        logit_scale_init: float = 1.0,
        mix_uniform_eps: float = 1e-4
    ):
        super().__init__()
        self.vfdim = vfdim
        self.cfdim = cfdim
        self.tau   = tau
        self.group_size = group_size
        self.group_dim = group_dim
        self.lambda_orth = lambda_orth
        self.activation = activation
        self.st_topk = st_topk
        self.clamp_negative = clamp_negative

        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32), requires_grad=True)

        self.cproj = nn.Linear(cfdim, vfdim)
        data = torch.load(config.save_cep_dir)
        concept_embs = data["concept_embs"].to(torch.float32)  # (concept_num, D)
        self.register_buffer("concept_embs", concept_embs)   # (C, cfdim)

        self.mu_head = nn.Linear(vfdim, group_dim)
        self.sigma_head = nn.Linear(vfdim, group_dim)

        assert activation == "softmax"
        self.softmax = StableSoftmax(dim=-1, mix_uniform_eps=mix_uniform_eps)

        self.last_topk_idx = None
        self.last_weights  = None

    def load_concept_embs(self, pt_path: str, device=None):
        data = torch.load(pt_path, map_location="cpu")
        concept_embs = data["concept_embs"].to(torch.float32)

        if device is None:
            if hasattr(self, "concept_embs"):
                device = self.concept_embs.device
            else:
                device = next(self.parameters()).device

        concept_embs = concept_embs.to(device)

        if concept_embs.shape[1] != self.cfdim:
            raise ValueError(
                f"New concept_embs dim mismatch: got {concept_embs.shape[1]}, expected {self.cfdim}"
            )

        self._buffers["concept_embs"] = concept_embs
        print(f"[MatchConcept] concept bank replaced: {pt_path}")
        print(f"[MatchConcept] new concept num = {concept_embs.shape[0]}, dim = {concept_embs.shape[1]}")

    def forward(self, matchF: torch.Tensor):
        """
        matchF: (B, vfdim)
        x:      (B, group_size, group_dim)
        return：
          cav:   (B, C)
          cep_x: (B, out_dim)
          orth_loss
        """
        B = matchF.shape[0]

        concept_proj = self.cproj(self.concept_embs)  # (C, vfdim)

        q = l2_normalize(matchF, dim=-1).float()     # (B, vfdim)
        k = l2_normalize(concept_proj, dim=-1).float()  # (C, vfdim)
        sims = q @ k.T                                 # (B, C)
        # sims = torch.einsum("bqf,cf->b,q,c", q, k)      # (B, C)
        if self.clamp_negative:
            sims = sims.clamp_min(0.0)

        tau   = max(float(self.tau), 1e-3)
        scale = self.logit_scale.clamp(0.1, 50.0)
        logits = (sims / tau) * scale                 # float32
        weights = self.softmax(logits)                # (B, C)
        return weights, concept_proj

def straight_through_topk(weights: torch.Tensor, k):
    """
    weights: [B, C]
    k:
        - int        -> [0, k)
        - tuple/list -> [start, end)
    return:
        weights_k: [B, C]
        topk_idx:  [B, chunk_size]
    """

    if isinstance(k, (tuple, list)):
        start, end = k
    else:
        start, end = 0, k
    assert end > start, "end should be > start"

    sorted_vals, sorted_idx = torch.sort(weights, dim=-1, descending=True)  # [B,C] each

    idx_chunk = sorted_idx[:, start:end]   # [B, end-start]
    mask = torch.zeros_like(weights).scatter(-1, idx_chunk, 1.0)  # [B,C]

    masked = weights * mask                          # [B,C]
    denom  = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    masked = masked / denom
    weights_k = (masked - weights).detach() + weights

    return weights_k, idx_chunk

class MultiTopkDistribution(nn.Module):
    def __init__(self, vfdim: int, group_size: int, group_dim: int):
        super().__init__()
        self.vfdim = vfdim
        self.group_size = group_size
        self.group_dim = group_dim

        self.mu_head = nn.Linear(vfdim, group_dim)
        self.sigma_head = nn.Linear(vfdim, group_dim)

    def forward(self, k, weights, concept_proj):
        weights_k, topk_idx = straight_through_topk(weights, k)  # <--- 用新的
        # print("Match cep idx k: ", topk_idx)

        z = weights_k @ concept_proj        # [B, vfdim]

        mu    = self.mu_head(z)             # [B, group_dim]
        sigma = F.softplus(self.sigma_head(z))  # [B, group_dim], >=0

        return mu, sigma

def build_chunk_sequence(klist):
    """
    klist: e.g.
      [1,2,3,4,5,6]          -> [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]
      [1,1,3,3,6,6]          -> [(0,1),(0,1),(1,3),(1,3),(3,6),(3,6)]
    """
    chunk_seq = []
    last_start = 0
    last_end   = klist[0]

    for i, k in enumerate(klist):
        if i == 0:
            last_start = 0
            last_end   = klist[0]
        else:
            if klist[i] != klist[i-1]:
                last_start = klist[i-1]
                last_end   = klist[i]
            # else: same as before, keep last_start/last_end
        chunk_seq.append( (last_start, last_end) )

    return chunk_seq

class Cepx2DirFusex(nn.Module):
    def __init__(self, group_size, group_dim):
        super().__init__()
        self.group_dim = group_dim
        self.group_size = group_size

        self.head_alpha = nn.Linear(group_dim, group_size*group_dim)
        self.head_beta = nn.Linear(group_dim, group_size*group_dim)
        self.head_gate = nn.Linear(group_dim, group_size*group_dim)

    def forward(self, x, cepx):

        b, g, d = x.shape

        alpha = self.head_alpha(cepx).view(b, g, d)
        beta = self.head_beta(cepx).view(b, g, d)
        gate = torch.sigmoid(self.head_gate(cepx)).view(b, g, d)
        xcepx = gate * x + (1 - gate) * (alpha * x + beta)
        return xcepx

class SPDiff(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        text_use_bert_cls = False,
        timesteps = config.timesteps,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__()
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        self.group1_size = config.group1_size
        self.group1_dim = config.group1_dim
        self.group2_size = config.group2_size
        self.group2_dim = config.group2_dim
        self.denoise_sample = config.denoise_sample

        self.match_mlp1 = nn.Sequential(
            nn.Linear(512 + 256, 512),
        )

        self.cep1 = MatchConcept(vfdim=512, cfdim=512, group_size=self.group1_size,
                                            group_dim=self.group1_dim)
        self.ditrib1 = MultiTopkDistribution(vfdim=512, group_size=self.group1_size,
                                            group_dim=self.group1_dim)
        self.match_mlp2 = nn.Sequential(
            nn.Linear(512 + 256, 512)
        )

        self.cep2 = MatchConcept(vfdim=512, cfdim=512, group_size=self.group2_size,
                                             group_dim=self.group2_dim)
        self.ditrib2 = MultiTopkDistribution(vfdim=512, group_size=self.group2_size,
                                             group_dim=self.group2_dim)
        self.fuse1 = Cepx2DirFusex(group_size=self.group1_size, group_dim=self.group1_dim)
        self.fuse2 = Cepx2DirFusex(group_size=self.group2_size, group_dim=self.group2_dim)
        self.conf_mlp1 = nn.Sequential(
            nn.Linear(self.group1_dim, self.group1_dim),
            nn.Softplus()
        )
        self.conf_mlp2 = nn.Sequential(
            nn.Linear(self.group2_dim, self.group2_dim),
            nn.Softplus()
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _dynamic_clip(self, x):
        s = 1.
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x, 'b ... -> b (...)').abs(),
                self.dynamic_thres_percentile,
                dim=-1
            )

            s.clamp_(min=1.)
            s = s.view(-1, *((1,) * (x.ndim - 1)))

        # clip by threshold, depending on whether static or dynamic
        x = x.clamp(-s, s) / s
        return x

    def single_topk_cepx(self, k, weights, concept_proj, c, t_emb, denoiseObj=None, denoiseF=None,
                        cep2distrF=None, fuse_module=None, conf_module=None):
        cep_mu, cep_sigma = cep2distrF(k, weights, concept_proj)
        xgens = []
        xcepxs = []
        for i in range(config.mc_sample):
            cep_x = torch.normal(cep_mu, cep_sigma).unsqueeze(1)
            xcepx = fuse_module(denoiseObj, cep_x)
            x_gen = denoiseF(denoiseObj, c, t_emb, xcepx)
            xgens.append(x_gen)
            xcepxs.append(xcepx)
        xgens = torch.stack(xgens, dim=0)  # mc_sample, b, g, d
        xcepxs = torch.stack(xcepxs, dim=0)  # mc_sample, b, g, d
        x_uncert = xcepxs.std(dim=0, unbiased=False)  # [B,G,D],
        x_conf = conf_module(x_uncert)
        x_gen = torch.mean(xgens, dim=0)
        xcepx = torch.mean(xcepxs, dim=0)
        return x_gen, x_conf, xcepx
    def p_mean_variance(self, x1, x2, t, k, clip_denoised: bool, cond_visual=None, cond_ratio=None, cond_scale=1.):
        c1, c2, t_emb = self.denoise_fn.CondTemb(t, cond_visual=cond_visual, cond_ratio=cond_ratio)
        matchF1 = torch.cat([c1, t_emb], dim=-1)
        matchF1 = self.match_mlp1(matchF1)

        matchF2 = torch.cat([c2, t_emb], dim=-1)
        matchF2 = self.match_mlp2(matchF2)

        match_weights1, concept_proj1 = self.cep1(matchF1)
        match_weights2, concept_proj2 = self.cep2(matchF2)
        noise1, conf1, cep_x1 = self.single_topk_cepx(k, match_weights1, concept_proj1, c1, t_emb,
                                                       denoiseObj=x1,
                                                       denoiseF=self.denoise_fn.group1_denoiser,
                                                       cep2distrF=self.ditrib1,
                                                       fuse_module=self.fuse1,
                                                       conf_module=self.conf_mlp1
                                                       )
        noise2, conf2, cep_x2 = self.single_topk_cepx(k, match_weights2, concept_proj2, c2, t_emb,
                                                       denoiseObj=x2,
                                                       denoiseF=self.denoise_fn.group2_denoiser,
                                                       cep2distrF=self.ditrib2,
                                                       fuse_module=self.fuse2,
                                                       conf_module=self.conf_mlp2
                                                       )

        x_recon1 = self.predict_start_from_noise(x1, t=t, noise=noise1)
        x_recon2 = self.predict_start_from_noise(x2, t=t, noise=noise2)

        if clip_denoised:
            x_recon1 = self._dynamic_clip(x_recon1)
            x_recon2 = self._dynamic_clip(x_recon2)

        model_mean1, posterior_variance1, posterior_log_variance1 = self.q_posterior(x_start=x_recon1, x_t=x1, t=t)
        model_mean2, posterior_variance2, posterior_log_variance2 = self.q_posterior(x_start=x_recon2, x_t=x2, t=t)
        return (model_mean1, posterior_variance1, posterior_log_variance1, conf1, cep_x1), \
               (model_mean2, posterior_variance2, posterior_log_variance2, conf2, cep_x2)

    @torch.inference_mode()
    def p_sample(self, x1, x2, t, k, cond_visual=None, cond_ratio=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x1.shape, x1.device
        (model_mean1, _, model_log_variance1, conf1, cep_x1), (
        model_mean2, _, model_log_variance2, conf2, cep_x2) = self.p_mean_variance(x1=x1,
                                                                           x2=x2,
                                                                           t=t, k=k,
                                                                           cond_visual=cond_visual,
                                                                           cond_ratio=cond_ratio,
                                                                           clip_denoised=clip_denoised,
                                                                           cond_scale=cond_scale)
        noise1 = torch.randn_like(x1)
        noise2 = torch.randn_like(x2)
        # no noise when t == 0
        nonzero_mask1 = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x1.shape) - 1)))
        nonzero_mask2 = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x2.shape) - 1)))
        return model_mean1 + nonzero_mask1 * (0.5 * model_log_variance1).exp() * noise1, conf1, cep_x1, \
               model_mean2 + nonzero_mask2 * (0.5 * model_log_variance2).exp() * noise2, conf2, cep_x2

    @torch.inference_mode()
    def p_sample_loop(self, shape1, shape2, k, cond_visual=None, cond_ratio=None, cond_scale=1.):
        device = self.betas.device

        b = shape1[0]
        img1 = torch.randn(shape1, device=device)
        img2 = torch.randn(shape2, device=device)
        conf1 = torch.randn(shape1, device=device)
        conf2 = torch.randn(shape2, device=device)
        denoiseObj1 = img1
        denoiseObj2 = img2
        for i in tqdm(range(config.timesteps - 1, -1, -1), desc='sampling loop time step'):
            img1, conf1, cep_x1, img2, conf2, cep_x2 = self.p_sample(img1, img2, torch.full((b,), i, device=device, dtype=torch.long),
                                                     k=k,
                                                     cond_visual=cond_visual, cond_ratio=cond_ratio,
                                                    cond_scale=cond_scale)
        return unnormalize_img(img1), conf1, cep_x1, denoiseObj1, unnormalize_img(img2), conf2, cep_x2, denoiseObj2

    @torch.inference_mode()
    def sample(self, shape1, shape2, cond_visual=None, cond_ratio = None, cond_scale=1., batch_size=16):
        device = next(self.denoise_fn.parameters()).device

        batch_size = cond_visual.shape[0] if exists(cond_visual) else batch_size
        # shape1=(batch_size, config.group1_size, config.group1_dim)
        return self.p_sample_loop(shape1, shape2, k=1,
                                  cond_visual=cond_visual, cond_ratio = cond_ratio,
                                  cond_scale=cond_scale)

    @torch.inference_mode()
    def multisample_topk_Noverlap(self, shape1, shape2, klist=None, cond_visual=None, cond_ratio=None, cond_scale=1.,
                         batch_size=16):
        device = next(self.denoise_fn.parameters()).device
        multiTrends_list1 = []
        multiTrends_list2 = []
        multiConfs_list1 = []
        multiConfs_list2 = []
        if klist is None:
            klist = [i + 1 for i in range(config.sample)]
        # 生成rank区间序列
        chunk_seq = build_chunk_sequence(klist)
        for i in range(config.sample):  # 采样次数
            k = chunk_seq[i]
            esti1, conf1, cep_x1, denoiseObj1, esti2, conf2, cep_x2, denoiseObj2 = self.p_sample_loop(shape1,
                                                                                                      shape2, k=k,
                                                                                                      cond_visual=cond_visual,
                                                                                                      cond_ratio=cond_ratio,
                                                                                                      cond_scale=cond_scale)
            multiTrends_list1.append(esti1)
            multiTrends_list2.append(esti2)
            multiConfs_list1.append(conf1)
            multiConfs_list2.append(conf2)
        return torch.stack(multiTrends_list1), torch.stack(multiConfs_list1), \
               torch.stack(multiTrends_list2), torch.stack(multiConfs_list2)  # (sample,B,1,3) and (sample,B,3,4)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def conf_gather(self, denoiseds, confs):
        s, b, g, d = denoiseds.shape
        '''g*d'''
        denoiseds = denoiseds.reshape(s, b, -1)  # s, b, g*d
        confs = confs.reshape(s, b, -1).sum(dim=-1)  # s, b, g, d---s, b, g*d---s, b
        conf_weight = F.softmax(confs, dim=0)  # s, b
        # print("conf_weight idx0: ", conf_weight[:, 0])
        denoised = (conf_weight.unsqueeze(-1) * denoiseds).sum(dim=0).reshape(b, g, d)  # b, g, d
        entropy_per_b = -(conf_weight * (conf_weight.clamp_min(1e-9).log())).sum(dim=0)  # [B]
        alpha = 0.6
        H0 = alpha * math.log(s)
        loss_entropy_mid = ((entropy_per_b - H0) ** 2).mean()
        return denoised, loss_entropy_mid

    def multi_topk_cepx_Noverlap(self, klist, weights, concept_proj, c, t_emb, denoiseObj=None, denoiseF=None,
                        cep2distrF=None, fuse_module=None, conf_module=None):
        multi_xgen = []
        multi_xconf = []
        multi_xcep = []
        # 生成rank区间序列
        chunk_seq = build_chunk_sequence(klist)
        for k_range in chunk_seq:
            cep_mu, cep_sigma = cep2distrF(k_range, weights, concept_proj)
            xgens = []
            xcepxs = []
            for i in range(config.mc_sample):
                cep_x = torch.normal(cep_mu, cep_sigma).unsqueeze(1)
                xcepx = fuse_module(denoiseObj, cep_x)
                x_gen = denoiseF(denoiseObj, c, t_emb, xcepx)
                xgens.append(x_gen)
                xcepxs.append(xcepx)
            xgens = torch.stack(xgens, dim=0)      # mc_sample, b, g, d
            xcepxs = torch.stack(xcepxs, dim=0)      # mc_sample, b, g, d
            x_uncert = xcepxs.std(dim=0, unbiased=False)  # [B,G,D],
            x_conf = conf_module(x_uncert)
            # print("x_conf[0]: ", x_conf[0, :, :])
            x_gen = torch.mean(xgens, dim=0)
            xcepx = torch.mean(xcepxs, dim=0)
            multi_xgen.append(x_gen)
            multi_xcep.append(xcepx)
            multi_xconf.append(x_conf)
        xgens = torch.stack(multi_xgen, dim=0)  # s, b, g, d
        xceps = torch.stack(multi_xcep, dim=0)  # s, b, g, d
        xconfs = torch.stack(multi_xconf, dim=0)
        return xgens, xconfs, xceps

    def p_mask_conf_gather(self, x_start1, x_start2, x_start2_label_mask, t, cond_visual = None, cond_ratio = None, noise = None, **kwargs):
        noise1 = default(noise, lambda: torch.randn_like(x_start1))
        noise2 = default(noise, lambda: torch.randn_like(x_start2))

        x_noisy1 = self.q_sample(x_start=x_start1, t=t, noise=noise1)
        x_noisy2 = self.q_sample(x_start=x_start2, t=t, noise=noise2)

        c1, c2, t_emb = self.denoise_fn.CondTemb(t, cond_visual=cond_visual, cond_ratio=cond_ratio, **kwargs)
        matchF1 = torch.cat([c1, t_emb], dim=-1)
        matchF1 = self.match_mlp1(matchF1)
        match_weights1, concept_proj1 = self.cep1(matchF1)

        matchF2 = torch.cat([c2, t_emb], dim=-1)
        matchF2 = self.match_mlp2(matchF2)
        match_weights2, concept_proj2 = self.cep2(matchF2)
        klist = [1, 2, 3, 4, 5]
        xgens1, xconfs1, xceps1 = self.multi_topk_cepx_Noverlap(klist, match_weights1, concept_proj1, c1, t_emb, denoiseObj=x_noisy1,
                                                  denoiseF=self.denoise_fn.group1_denoiser,
                                                  cep2distrF = self.ditrib1,
                                                  fuse_module=self.fuse1,
                                                  conf_module=self.conf_mlp1
                                                  )
        xgens2, xconfs2, xceps2 = self.multi_topk_cepx_Noverlap(klist, match_weights2, concept_proj2, c2, t_emb, denoiseObj=x_noisy2,
                                                  denoiseF=self.denoise_fn.group2_denoiser,
                                                  cep2distrF = self.ditrib2,
                                                  fuse_module=self.fuse2,
                                                  conf_module=self.conf_mlp2
                                                  )

        denoised1, loss_entropy_mid1 = self.conf_gather(xgens1, xconfs1)       # b, 1, 3
        denoised2, loss_entropy_mid2 = self.conf_gather(xgens2, xconfs2)       # b, 4, 3
        print("loss_entropy_mid1 = {}, loss_entropy_mid2 = {}".format(loss_entropy_mid1, loss_entropy_mid2))
        if torch.isnan(xgens1).any() or torch.isnan(xgens2).any():
            raise ValueError("contains NaN")
        if self.loss_type == 'l1':
            noise_loss1 = ((denoised1 - noise1).abs()).mean(0).sum()
            noise_loss2 = ((denoised2 - noise2).abs() * x_start2_label_mask).mean(0).sum()
        elif self.loss_type == 'l2':
            noise_loss1 = (((denoised1 - noise1).abs() ** 2)).mean(0).sum()
            noise_loss2 = (((denoised2 - noise2).abs() ** 2) * x_start2_label_mask).mean(0).sum()
        else:
            raise NotImplementedError()
        loss = [noise_loss1+loss_entropy_mid1, noise_loss2+loss_entropy_mid2]
        print("deter tasks noise loss = {}, other tasks noise loss = {}".format(noise_loss1, noise_loss2))
        return loss

    def forward(self, x1, x2, x2_label_mask, *args, **kwargs):
        b, device = x1.shape[0], x1.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x1 = normalize_img(x1)
        x2 = normalize_img(x2)
        return self.p_mask_conf_gather(x1, x2, x2_label_mask, t, *args, **kwargs)
