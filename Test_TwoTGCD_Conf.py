import os
import numpy as np
import torchvision
import torch
import env.Config as config
import torch.nn.functional as F
import torch.optim
from torch import nn
from tqdm import tqdm
from env.Dataset import *
import random

def compute_loss(y_est, y_true, mask):
    abs_error = torch.abs(y_est - y_true)  # [B, num_vars]
    masked_error = abs_error * mask
    loss = masked_error.sum() / mask.sum().clamp(min=1e-6)  # Prevent division by zero
    print("esti_loss={}".format(loss))
    return loss

def AllWeightGatherConf(gen_tca_all, gen_conf_all):
    """
    gen_tca_all: (S, B, G, Q)
    gen_conf_all: (S, B, G, Q)
    """
    S, B, G, Q = gen_tca_all.shape
    gen_tca_all = gen_tca_all.reshape(S, B, -1)  # s, b, g*d
    confs = gen_conf_all.reshape(S, B, -1).sum(dim=-1)  # s, b, g, d---s, b, g*d---s, b
    conf_weight = F.softmax(confs, dim=0)  # s, b
    # conf_weight = 1 - F.softmax(confs, dim=0)  # s, b
    # conf_weight = torch.softmax(confs / 0.07, dim=0)  # (S,B)
    print("conf_weight idx0: ", conf_weight[:, 0])
    gen_tca = (conf_weight.unsqueeze(-1) * gen_tca_all).sum(dim=0).reshape(B, G, Q)  # b, g, d
    return gen_tca.reshape(B, -1)

def test_multiconf(model, dataset):
    model.eval()
    '''quad-tg'''
    gen_var_names = [
        "r34ne", "r50ne", "r64ne",
        "r34se", "r50se", "r64se",
        "r34sw", "r50sw", "r64sw",
        "r34nw", "r50nw", "r64nw"
    ]
    gen_var_ranges = {
        "r34ne": (10, 390), "r50ne": (7, 231), "r64ne": (2, 123),
        "r34se": (10, 430), "r50se": (7, 215), "r64se": (5, 135),
        "r34sw": (5, 460), "r50sw": (5, 205), "r64sw": (5, 120),
        "r34nw": (10, 445), "r50nw": (5, 238), "r64nw": (2, 124)
    }
    deter_var_names = [
        "msw", "mslp", "rmw"
    ]
    deter_var_ranges = {
        "msw": (35, 170),
        "mslp": (882, 1008),
        "rmw": (5, 130)
    }
    group_vars = {
        "r34": ["r34ne", "r34se", "r34sw", "r34nw"],
        "r50": ["r50ne", "r50se", "r50sw", "r50nw"],
        "r64": ["r64ne", "r64se", "r64sw", "r64nw"]
    }
    group_error = {k: 0.0 for k in group_vars}
    group_count = {k: 0 for k in group_vars}
    gen_total_error = {name: 0.0 for name in gen_var_names}
    deter_total_error = {name: 0.0 for name in deter_var_names}
    gen_valid_count = {name: 0 for name in gen_var_names}
    deter_valid_count = {name: 0 for name in deter_var_names}
    loss = 0
    batch_num = 0
    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataset)):
            k8_btemp = data["k8_btemp"].to(config.device)
            pxh_k8_btemp = data["pxh_k8_btemp"].to(config.device)
            btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)
            pr = data['pr'].to(config.device)

            group1_label = data["group1_label"].to(config.device)  # b, 1, 3
            group2_label = data["group2_label"].to(config.device)  # b, 4, 3
            group2_label_mask = data["group2_label_mask"].to(config.device)

            B = group1_label.shape[0]
            shape1 = (B, config.group1_size, config.group1_dim)
            shape2 = (B, config.group2_size, config.group2_dim)
            klist = None
            # best model
            gen_tca_all1, gen_conf_all1, gen_tca_all2, gen_conf_all2 = \
                model.multisample_topk_Noverlap(shape1, shape2, klist=klist, cond_visual=btemp, cond_ratio=pr)

            S, B, G1, Q1 = gen_tca_all1.shape  # G=grouped tasks, Q=dimensions per group
            S, B, G2, Q2 = gen_tca_all2.shape  # G=grouped tasks, Q=dimensions per group

            label1_flat = group1_label.reshape(B, -1)  # (B,3)
            label2_flat = group2_label.reshape(B, -1)  # (B,12)
            mask2_flat = group2_label_mask.reshape(B, -1)  # (B,12)

            gen_tca1 = AllWeightGatherConf(gen_tca_all1, gen_conf_all1)
            gen_tca2 = AllWeightGatherConf(gen_tca_all2, gen_conf_all2)

            generate_loss = compute_loss(gen_tca2, label2_flat, mask2_flat)
            deter_loss = F.l1_loss(gen_tca1, label1_flat)
            print("batch generate_loss = {}, deter_loss = {} ".format(generate_loss, deter_loss))
            loss += (generate_loss + deter_loss)
            batch_num += 1

            for idx, var in enumerate(deter_var_names):
                min_val, max_val = deter_var_ranges[var]
                pred = gen_tca1[:, idx].cpu().numpy() * (max_val - min_val) + min_val
                gt = label1_flat[:, idx].cpu().numpy() * (max_val - min_val) + min_val
                deter_total_error[var] += np.sum(np.abs(pred - gt))
                deter_valid_count[var] += len(pred)
            for idx, var in enumerate(gen_var_names):
                min_val, max_val = gen_var_ranges[var]
                mask = mask2_flat[:, idx] == 1
                if mask.sum().item() == 0:
                    continue
                pred = gen_tca2[:, idx][mask].cpu().numpy() * (max_val - min_val) + min_val
                gt = label2_flat[:, idx][mask].cpu().numpy() * (max_val - min_val) + min_val
                gen_total_error[var] += np.sum(np.abs(pred - gt))
                gen_valid_count[var] += len(pred)

            for group_name, vars_in_group in group_vars.items():
                indices = [gen_var_names.index(var) for var in vars_in_group]
                group_preds = []
                group_gts = []

                for b in range(B):
                    valid_quadrants = []
                    for i in indices:
                        if mask2_flat[b, i] == 1:
                            min_val, max_val = gen_var_ranges[gen_var_names[i]]
                            pred_val = gen_tca2[b, i].item() * (max_val - min_val) + min_val
                            gt_val = label2_flat[b, i].item() * (max_val - min_val) + min_val
                            valid_quadrants.append((pred_val, gt_val))
                    if len(valid_quadrants) > 0:
                        pred_avg = sum([p for p, _ in valid_quadrants]) / len(valid_quadrants)
                        gt_avg = sum([g for _, g in valid_quadrants]) / len(valid_quadrants)
                        group_preds.append(pred_avg)
                        group_gts.append(gt_avg)

                if len(group_preds) > 0:
                    group_error[group_name] += np.sum(np.abs(np.array(group_preds) - np.array(group_gts)))
                    group_count[group_name] += len(group_preds)

        print("\n[Test Results with Mask]")
        for var in deter_var_names:
            if deter_valid_count[var] == 0:
                print(f"[{var}] No valid samples.")
                continue
            avg_mae = deter_total_error[var] / deter_valid_count[var]
            print(f"[Test MAE] {var}: {avg_mae:.4f} (Valid N={deter_valid_count[var]})")
        for var in gen_var_names:
            if gen_valid_count[var] == 0:
                print(f"[{var}] No valid samples.")
                continue
            avg_mae = gen_total_error[var] / gen_valid_count[var]
            print(f"[Test MAE] {var}: {avg_mae:.4f} (Valid N={gen_valid_count[var]})")

    print("test mean loss = {} ".format(loss / batch_num))
    print("\n[Quadrant-Averaged Test MAE]")
    for g in group_vars.keys():
        if group_count[g] == 0:
            print(f"[{g}] No valid samples.")
            continue
        avg_mae = group_error[g] / group_count[g]
        print(f"[Test MAE] {g}_avg: {avg_mae:.4f} (Valid N={group_count[g]})")
    return

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
    test_dataset = TCDataset_HisPR_TwoTG(config.predict_k8_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )
    DDPM = torch.load(config.predict_model)

    test_multiconf(DDPM, test_dataloader)
