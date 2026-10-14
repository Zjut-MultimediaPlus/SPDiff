import numpy as np
import torch
import os
from tqdm import tqdm
import env.Config as config
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

def rotate_quadrant_labels(labels, filename):
    """Align NE/SE/SW/NW targets with clockwise image rotation (north up, east right).

    Each quadrant contains R34, R50 and R64. Apply before normalization and
    mask construction. Scalar intensity targets and historical ratios are invariant.
    """
    tag = os.path.splitext(filename)[0].split('_')[-1]
    if not tag.startswith('rotate'):
        return labels
    if tag not in ('rotate90', 'rotate180', 'rotate270'):
        raise ValueError(f'Unsupported quadrant rotation: {filename}; only 90/180/270 are supported.')
    steps = int(tag[6:]) // 90
    quadrants = np.asarray(labels).reshape(4, 3)
    return np.roll(quadrants, steps, axis=0).reshape(-1).tolist()

def find_previous_hours(time_str, x):
    time_format = "%Y%m%d%H"
    time = datetime.strptime(time_str, time_format)
    previous_time = time - timedelta(hours=x)
    previous_time_str = previous_time.strftime(time_format)

    return previous_time_str

def get_previous_npy_index(npy_index, x):
    # npy_index
    parts = npy_index.split('_')
    year_typhoon_name = parts[0] + '_' + parts[1]
    time_str = parts[2]  

    time_format = '%Y%m%d%H'
    time_obj = datetime.strptime(time_str, time_format)

    previous_time_obj = time_obj - timedelta(hours=x)
    previous_time_str = previous_time_obj.strftime(time_format)
    previous_npy_index = year_typhoon_name + "_" + previous_time_str

    return previous_npy_index

def default_loader(path):
    raw_data = np.load(path, allow_pickle=True)
    tensor_data = torch.from_numpy(raw_data)
    tensor_data = tensor_data.type(torch.FloatTensor)
    return tensor_data

def load_dict_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_fnorm_now_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        maxv = statistic_dic['now_train'][c][0]
        minv = statistic_dic['now_train'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)

    return x_norm

def get_fnorm_p3h_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        maxv = statistic_dic['p3h_train'][c][0]
        minv = statistic_dic['p3h_train'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)
    return x_norm

class PDTCDataset():
    def __init__(self, k8_path):

        self.k8_paths = k8_path
        self.k8_btemps = []
        self.pxh_k8_btemps = []

        self.masks = []
        self.msws = []
        self.mslps = []
        self.rmws = []
        self.r34nes = []
        self.r34ses = []
        self.r34sws = []
        self.r34nws = []
        self.r50nes = []
        self.r50ses = []
        self.r50sws = []
        self.r50nws = []
        self.r64nes = []
        self.r64ses = []
        self.r64sws = []
        self.r64nws = []

        self.lats = []
        self.lons = []
        self.ts = []
        self.levels = []
        self.plevels = []

        self.pre_tcfs = []
        self.pre_isrs = []
        self.pre_pwrs = []
        self.pre_prrs = []
        self.pre_rrrs = []
        self.pre_levels = []
        self.norm_ts = []
        self.sample_keys = []


        self.labels_dic = load_dict_from_pickle(config.labels_path)

        k8_files = sorted(name for name in os.listdir(k8_path) if name.endswith('.npy'))
        k8_files_set = set(k8_files)

        self.k8_sta_dic = load_dict_from_pickle(config.k8_sta_path)
        self.level_name = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']

        for i, filename in enumerate(k8_files):
            fname_split = filename.split("_")
            tcname = fname_split[1]
            year = fname_split[0]

            # e.g. 2023_BOLAVEN_2023100809.npy
            if len(fname_split) == 3:
                isotime = fname_split[2][:-4]
            else:
                isotime = fname_split[2]
            
            labels_dic_key = year + "_" + tcname + "_" + isotime

            # e.g. BOLAVEN_2023100806.npy
            p3h_labels_dic_key = get_previous_npy_index(labels_dic_key, 3)
            if len(fname_split) == 3:
                pxh_k8_fname = p3h_labels_dic_key + ".npy"
            else:
                pxh_k8_fname = p3h_labels_dic_key + "_" + fname_split[3]

            if filename in self.k8_sta_dic['now_inval_record'] or pxh_k8_fname in self.k8_sta_dic['p3h_inval_record']:
                continue
            if pxh_k8_fname not in k8_files_set:
                continue
            if labels_dic_key not in self.labels_dic.keys():
                continue

            iso_time, lat, lon, t, level, mslp, msw, rmw, r34, r50, \
            r34ne, r34se, r34sw, r34nw, \
            r50ne, r50se, r50sw, r50nw, \
            r64ne, r64se, r64sw, r64nw = self.labels_dic[labels_dic_key]

            labels = [
                r34ne, r50ne, r64ne,
                r34se, r50se, r64se,
                r34sw, r50sw, r64sw,
                r34nw, r50nw, r64nw
            ]
            labels = rotate_quadrant_labels(labels, filename)
            (r34ne, r50ne, r64ne,
             r34se, r50se, r64se,
             r34sw, r50sw, r64sw,
             r34nw, r50nw, r64nw) = labels
            mask = [int(v != 0) for v in labels]
            mask = torch.tensor(mask, dtype=torch.long)

            his_dic_index = get_previous_npy_index(labels_dic_key, 12)
            if his_dic_index not in self.labels_dic.keys():
                continue

            his_level, his_mslp, his_msw, his_rmw, his_r34, his_r50, \
            his_r34ne, his_r34se, his_r34sw, his_r34nw, \
            his_r50ne, his_r50se, his_r50sw, his_r50nw, \
            his_r64ne, his_r64se, his_r64sw, his_r64nw = \
                self.labels_dic[his_dic_index][4:]

            if his_r34 == 0 or his_rmw == 0 or his_r50 == 0 or his_msw == 0:
                continue
            if his_rmw > his_r34:
                continue

            his_tcf = 1 - (his_rmw / his_r34)
            his_isr = his_msw / his_rmw
            his_pwr = his_mslp / his_msw
            his_prr = his_mslp / his_r34
            his_rrr = 1 - (his_r50 / his_r34)


            self.pre_tcfs.append(his_tcf)
            self.pre_isrs.append(his_isr)
            self.pre_pwrs.append(his_pwr)
            self.pre_prrs.append(his_prr)
            self.pre_rrrs.append(his_rrr)
            self.plevels.append(his_level)
            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.levels.append(level)
            self.norm_ts.append(t)

            self.pxh_k8_btemps.append(os.path.join(k8_path, pxh_k8_fname))
            self.k8_btemps.append(os.path.join(k8_path, filename))

            self.masks.append(mask)
            self.msws.append(msw)
            self.mslps.append(mslp)
            self.rmws.append(rmw)
            self.r34nes.append(r34ne)
            self.r34ses.append(r34se)
            self.r34sws.append(r34sw)
            self.r34nws.append(r34nw)
            self.r50nes.append(r50ne)
            self.r50ses.append(r50se)
            self.r50sws.append(r50sw)
            self.r50nws.append(r50nw)
            self.r64nes.append(r64ne)
            self.r64ses.append(r64se)
            self.r64sws.append(r64sw)
            self.r64nws.append(r64nw)
            self.sample_keys.append(labels_dic_key)


        if not self.msws:
            raise ValueError(f'No usable samples in {k8_path}; check labels, history and split contents.')
        print("msws max = {}, min = {}".format(max(self.msws), min(self.msws)))
        print("mslps max = {}, min = {}".format(max(self.mslps), min(self.mslps)))
        print("rmws max = {}, min = {}".format(max(self.rmws), min(filter(lambda x: x != 0, self.rmws))))
        print("r34nes max = {}, min = {}".format(max(self.r34nes), min(filter(lambda x: x != 0, self.r34nes))))
        print("r34ses max = {}, min = {}".format(max(self.r34ses), min(filter(lambda x: x != 0, self.r34ses))))
        print("r34sws max = {}, min = {}".format(max(self.r34sws), min(filter(lambda x: x != 0, self.r34sws))))
        print("r34nws max = {}, min = {}".format(max(self.r34nws), min(filter(lambda x: x != 0, self.r34nws))))
        print("r50nes max = {}, min = {}".format(max(self.r50nes), min(filter(lambda x: x != 0, self.r50nes))))
        print("r50ses max = {}, min = {}".format(max(self.r50ses), min(filter(lambda x: x != 0, self.r50ses))))
        print("r50sws max = {}, min = {}".format(max(self.r50sws), min(filter(lambda x: x != 0, self.r50sws))))
        print("r50nws max = {}, min = {}".format(max(self.r50nws), min(filter(lambda x: x != 0, self.r50nws))))
        print("r64nes max = {}, min = {}".format(max(self.r64nes), min(filter(lambda x: x != 0, self.r64nes))))
        print("r64ses max = {}, min = {}".format(max(self.r64ses), min(filter(lambda x: x != 0, self.r64ses))))
        print("r64sws max = {}, min = {}".format(max(self.r64sws), min(filter(lambda x: x != 0, self.r64sws))))
        print("r64nws max = {}, min = {}".format(max(self.r64nws), min(filter(lambda x: x != 0, self.r64nws))))
        print("t max = {}, min = {}".format(max(self.ts), min(self.ts)))
        print("levels max = {}, min = {}".format(max(self.levels), min(self.levels)))
        print("pre_tcf max = {}, min = {}".format(max(self.pre_tcfs), min(self.pre_tcfs)))
        print("pre_isr max = {}, min = {}".format(max(self.pre_isrs), min(self.pre_isrs)))
        print("pre_pwr max = {}, min = {}".format(max(self.pre_pwrs), min(self.pre_pwrs)))
        print("pre_prr max = {}, min = {}".format(max(self.pre_prrs), min(self.pre_prrs)))
        print("pre_rrr max = {}, min = {}".format(max(self.pre_rrrs), min(self.pre_rrrs)))

        for i in range(len(self.msws)):
            self.msws[i] = (self.msws[i] - 35) / (170 - 35)
            self.mslps[i] = (self.mslps[i] - 882) / (1008 - 882)
            self.rmws[i] = (self.rmws[i] - 5) / (130 - 5)

            self.r34nes[i] = (self.r34nes[i] - 10) / (390 - 10)
            self.r34ses[i] = (self.r34ses[i] - 10) / (430 - 10)
            self.r34sws[i] = (self.r34sws[i] - 5) / (460 - 5)
            self.r34nws[i] = (self.r34nws[i] - 10) / (445 - 10)

            self.r50nes[i] = (self.r50nes[i] - 7) / (231 - 7)
            self.r50ses[i] = (self.r50ses[i] - 7) / (215 - 7)
            self.r50sws[i] = (self.r50sws[i] - 5) / (205 - 5)
            self.r50nws[i] = (self.r50nws[i] - 5) / (238 - 5)

            self.r64nes[i] = (self.r64nes[i] - 2) / (123 - 2)
            self.r64ses[i] = (self.r64ses[i] - 5) / (135 - 5)
            self.r64sws[i] = (self.r64sws[i] - 5) / (120 - 5)
            self.r64nws[i] = (self.r64nws[i] - 2) / (124 - 2)

            self.pre_isrs[i] = (self.pre_isrs[i] - 0.38) / (34 - 0.38)
            self.pre_pwrs[i] = (self.pre_pwrs[i] - 5.18) / (20.02 - 5.18)
            self.pre_prrs[i] = (self.pre_prrs[i] - 2.76) / (60.99 - 2.76)

    def __len__(self):
        return len(self.k8_btemps)

    def __getitem__(self, index):
        btemp_file_path = self.k8_btemps[index]
        k8_btemp = default_loader(btemp_file_path)
        k8_btemp = get_fnorm_now_chw(k8_btemp, self.k8_sta_dic)

        pxh_btemp_file_path = self.pxh_k8_btemps[index]
        pxh_k8_btemp = default_loader(pxh_btemp_file_path)
        pxh_k8_btemp = get_fnorm_p3h_chw(pxh_k8_btemp, self.k8_sta_dic)

        msw = self.msws[index]
        mslp = self.mslps[index]
        rmw = self.rmws[index]
        r34ne = self.r34nes[index]
        r34se = self.r34ses[index]
        r34sw = self.r34sws[index]
        r34nw = self.r34nws[index]
        r50ne = self.r50nes[index]
        r50se = self.r50ses[index]
        r50sw = self.r50sws[index]
        r50nw = self.r50nws[index]
        r64ne = self.r64nes[index]
        r64se = self.r64ses[index]
        r64sw = self.r64sws[index]
        r64nw = self.r64nws[index]

        group1_label = torch.tensor([msw, mslp, rmw], dtype=torch.float32).reshape(1, 3)
        group2_label = torch.tensor([
            r34ne, r50ne, r64ne,
            r34se, r50se, r64se,
            r34sw, r50sw, r64sw,
            r34nw, r50nw, r64nw
        ], dtype=torch.float32).reshape(4, 3)

        group2_label_mask = self.masks[index].reshape(4, 3)

        level = self.levels[index]
        lat = self.lats[index]
        lon = self.lons[index]
        pre_tcf = round(self.pre_tcfs[index], 3)
        pre_isr = round(self.pre_isrs[index], 3)
        pre_pwr = round(self.pre_pwrs[index], 3)
        pre_prr = round(self.pre_prrs[index], 3)
        pre_rrr = round(self.pre_rrrs[index], 3)
        t = self.ts[index]
        pr_list = [pre_tcf, pre_isr, pre_pwr, pre_prr, pre_rrr]
        pr = torch.tensor(pr_list, dtype=torch.float32)
        sample_key = self.sample_keys[index]

        sample = {
            'lat': lat,
            'lon': lon,
            'occur_t': t,
            'pre_tcf': pre_tcf,
            'group1_label': group1_label,
            'level': level,
            'group2_label': group2_label,
            'group2_label_mask': group2_label_mask,
            'k8_btemp': k8_btemp,
            'pxh_k8_btemp': pxh_k8_btemp,
            'pr': pr,
            'sample_key': sample_key,
            }
        return sample







# SETCD-Sat implementation
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import torch

import env.Config as config




def setcd_get_previous_npy_index(npy_index, x):
    """Return the filename key for the same cyclone a given number of hours earlier."""
    parts = npy_index.split('_')
    if len(parts) < 3:
        raise ValueError(f"Bad npy_index format: {npy_index}")

    year_typhoon_name = "_".join(parts[:-1])
    time_str = parts[-1]

    if len(time_str) not in (10, 12):
        raise ValueError(f'Expected YYYYMMDDHH or YYYYMMDDHHMM timestamp: {time_str}')
    time_format = '%Y%m%d%H' if len(time_str) == 10 else '%Y%m%d%H%M'
    time_obj = datetime.strptime(time_str, time_format)
    previous_time_obj = time_obj - timedelta(hours=x)
    previous_time_str = previous_time_obj.strftime(time_format)

    previous_npy_index = year_typhoon_name + "_" + previous_time_str
    return previous_npy_index


def setcd_center_crop_2d(x: torch.Tensor, out_h: int = 156, out_w: int = 156) -> torch.Tensor:
    """
    Center crop for 2D or batched/channel 2D tensors.
    Supports shapes: (H,W), (C,H,W), (N,C,H,W), ...
    """
    H, W = x.shape[-2], x.shape[-1]
    if out_h > H or out_w > W:
        raise ValueError(f"Output size ({out_h},{out_w}) must be <= input size ({H},{W}).")

    top = (H - out_h) // 2
    left = (W - out_w) // 2
    return x[..., top:top + out_h, left:left + out_w]


def setcd_default_loader(path):
    raw_data = np.load(path, allow_pickle=True)[0]
    tensor_data = torch.from_numpy(raw_data)
    btemp = tensor_data.type(torch.FloatTensor)
    btemp = setcd_center_crop_2d(btemp, 156, 156)
    return btemp.unsqueeze(0) if btemp.ndim == 2 else btemp


def setcd_load_dict_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def setcd__safe_float(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return default
        try:
            return float(x)
        except Exception:
            return default
    if isinstance(x, (int, np.integer, float, np.floating)):
        try:
            if isinstance(x, float) and np.isnan(x):
                return default
            return float(x)
        except Exception:
            return default
    try:
        return float(x)
    except Exception:
        return default


def setcd__value_exists(v):
    """A label is valid when its numeric value is greater than zero."""
    val = setcd__safe_float(v, default=0.0)
    return np.isfinite(val) and val > 0


def setcd__safe_mean_of_valid(vals):
    valid_vals = [setcd__safe_float(v, 0.0) for v in vals if setcd__value_exists(v)]
    if len(valid_vals) == 0:
        return 0.0
    return float(sum(valid_vals) / len(valid_vals))


def setcd__safe_min_nonzero(arr):
    arr2 = [x for x in arr if x != 0]
    if len(arr2) == 0:
        return 0
    return min(arr2)


def setcd__msw_to_level(msw):
    """Derive the TD, TS, H1, H2, H3, H4 or H5 category from wind speed."""
    msw = setcd__safe_float(msw, 0.0)
    if msw < 34:
        return 0
    elif msw < 64:
        return 1
    elif msw < 83:
        return 2
    elif msw < 96:
        return 3
    elif msw < 113:
        return 4
    elif msw < 137:
        return 5
    else:
        return 6


def setcd__parse_filename(filename):
    """Parse observation filenames and optional rotation suffixes; cyclone names may contain underscores."""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")

    if len(parts) < 3:
        return None

    aug_tag = None
    if parts[-1] in ("rotate90", "rotate180", "rotate270"):
        aug_tag = parts[-1]
        iso_time = parts[-2]
        year = parts[0]
        tcname = "_".join(parts[1:-2])
    else:
        iso_time = parts[-1]
        year = parts[0]
        tcname = "_".join(parts[1:-1])

    if tcname == "":
        return None

    base_key = f"{year}_{tcname}_{iso_time}"
    return {
        "year": year,
        "tcname": tcname,
        "iso_time": iso_time,
        "aug_tag": aug_tag,
        "base_key": base_key,
        "stem": stem,
    }


class SETCDSatDataset():
    def __init__(self, k8_path):
        self.k8_paths = k8_path
        self.k8_btemps = []
        self.pxh_k8_btemps = []

        self.masks = []
        self.msws = []
        self.mslps = []
        self.rmws = []
        self.r34nes = []
        self.r34ses = []
        self.r34sws = []
        self.r34nws = []
        self.r50nes = []
        self.r50ses = []
        self.r50sws = []
        self.r50nws = []
        self.r64nes = []
        self.r64ses = []
        self.r64sws = []
        self.r64nws = []

        self.lats = []
        self.lons = []
        self.ts = []
        self.levels = []
        self.plevels = []

        self.pre_tcfs = []
        self.pre_isrs = []
        self.pre_pwrs = []
        self.pre_prrs = []
        self.pre_rrrs = []
        self.pre_levels = []
        self.norm_ts = []
        self.sample_keys = []


        self.level_name = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']

        row_dict_path = getattr(config, "ibtracs_row_dict_path", None)
        if row_dict_path is None:
            row_dict_path = getattr(config, "labels_path", None)

        if row_dict_path is None or (not os.path.exists(row_dict_path)):
            raise FileNotFoundError("not find ibtracs_row_dict.pkl")
        self.labels_dic = setcd_load_dict_from_pickle(row_dict_path)

        columns_path = getattr(config, "ibtracs_columns_path", None)
        if columns_path is None or (not os.path.exists(columns_path)):
            raise FileNotFoundError("not find ibtracs_columns.pkl")
        self.columns = setcd_load_dict_from_pickle(columns_path)
        self.col2idx = {c: i for i, c in enumerate(self.columns)}

        age_pkl_path = getattr(config, "ibtracs_age_hours_path", None)
        if age_pkl_path is None:
            age_pkl_path = getattr(config, "ibtracs_age_pkl_path", None)
        if age_pkl_path is None or (not os.path.exists(age_pkl_path)):
            raise FileNotFoundError(
                "not find age pkl, please set config.ibtracs_age_hours_path"
            )
        self.age_dic = setcd_load_dict_from_pickle(age_pkl_path)

        required_cols = [
            "USA_WIND", "USA_PRES", "USA_RMW",
            "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
            "USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW",
            "USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW",
        ]
        for c in required_cols:
            if c not in self.col2idx:
                raise KeyError(f"ibtracs_columns.pkl lack col: {c}")

        self.idx_lat = self.col2idx.get("USA_LAT", None)
        self.idx_lon = self.col2idx.get("USA_LON", None)
        self.idx_r34 = self.col2idx.get("USA_R34", None)
        self.idx_r50 = self.col2idx.get("USA_R50", None)
        self.idx_level = self.col2idx.get("LEVEL", None)
        if self.idx_level is None:
            self.idx_level = self.col2idx.get("USA_SSHS", None)

        nan_half_path = getattr(config, "nan_over_half_first_channel_path", None)
        if nan_half_path is not None and os.path.exists(nan_half_path):
            nan_half_obj = setcd_load_dict_from_pickle(nan_half_path)
            self.nan_over_half_set = set(nan_half_obj)
        else:
            self.nan_over_half_set = set()

        k8_files = sorted([f for f in os.listdir(k8_path) if f.endswith(".npy")])
        k8_files_set = set(k8_files)

        for _, filename in enumerate(k8_files):
            parsed = setcd__parse_filename(filename)
            if parsed is None:
                continue

            labels_dic_key = parsed["base_key"]

            if labels_dic_key in self.nan_over_half_set:
                continue

            prev6h_labels_dic_key = setcd_get_previous_npy_index(labels_dic_key, 6)
            if parsed["aug_tag"] is None:
                pxh_k8_fname = prev6h_labels_dic_key + ".npy"
            else:
                pxh_k8_fname = prev6h_labels_dic_key + "_" + parsed["aug_tag"] + ".npy"

            if pxh_k8_fname not in k8_files_set:
                continue

            if prev6h_labels_dic_key in self.nan_over_half_set:
                continue

            if labels_dic_key not in self.labels_dic:
                continue

            # Read storm age from the supplied metadata.
            if labels_dic_key not in self.age_dic:
                continue

            row = self.labels_dic[labels_dic_key]

            msw = setcd__safe_float(row[self.col2idx["USA_WIND"]], 0.0)
            mslp = setcd__safe_float(row[self.col2idx["USA_PRES"]], 0.0)
            rmw = setcd__safe_float(row[self.col2idx["USA_RMW"]], 0.0)
            # Group-one targets have no loss mask: retain complete valid targets only.
            if not all(setcd__value_exists(v) for v in (msw, mslp, rmw)):
                continue

            r34ne = setcd__safe_float(row[self.col2idx["USA_R34_NE"]], 0.0)
            r34se = setcd__safe_float(row[self.col2idx["USA_R34_SE"]], 0.0)
            r34sw = setcd__safe_float(row[self.col2idx["USA_R34_SW"]], 0.0)
            r34nw = setcd__safe_float(row[self.col2idx["USA_R34_NW"]], 0.0)

            r50ne = setcd__safe_float(row[self.col2idx["USA_R50_NE"]], 0.0)
            r50se = setcd__safe_float(row[self.col2idx["USA_R50_SE"]], 0.0)
            r50sw = setcd__safe_float(row[self.col2idx["USA_R50_SW"]], 0.0)
            r50nw = setcd__safe_float(row[self.col2idx["USA_R50_NW"]], 0.0)

            r64ne = setcd__safe_float(row[self.col2idx["USA_R64_NE"]], 0.0)
            r64se = setcd__safe_float(row[self.col2idx["USA_R64_SE"]], 0.0)
            r64sw = setcd__safe_float(row[self.col2idx["USA_R64_SW"]], 0.0)
            r64nw = setcd__safe_float(row[self.col2idx["USA_R64_NW"]], 0.0)

            labels = [
                r34ne, r50ne, r64ne,
                r34se, r50se, r64se,
                r34sw, r50sw, r64sw,
                r34nw, r50nw, r64nw
            ]
            labels = rotate_quadrant_labels(labels, filename)
            (r34ne, r50ne, r64ne,
             r34se, r50se, r64se,
             r34sw, r50sw, r64sw,
             r34nw, r50nw, r64nw) = labels
            mask = [int(setcd__value_exists(v)) for v in labels]
            mask = torch.tensor(mask, dtype=torch.long)

            lat = setcd__safe_float(row[self.idx_lat], 0.0) if self.idx_lat is not None else 0.0
            lon = setcd__safe_float(row[self.idx_lon], 0.0) if self.idx_lon is not None else 0.0

            if self.idx_level is not None:
                level = int(setcd__safe_float(row[self.idx_level], setcd__msw_to_level(msw)))
            else:
                level = setcd__msw_to_level(msw)

            t = int(setcd__safe_float(self.age_dic[labels_dic_key], 0.0))

            his_dic_index = setcd_get_previous_npy_index(labels_dic_key, 12)
            if his_dic_index not in self.labels_dic:
                continue

            his_row = self.labels_dic[his_dic_index]

            his_msw = setcd__safe_float(his_row[self.col2idx["USA_WIND"]], 0.0)
            his_mslp = setcd__safe_float(his_row[self.col2idx["USA_PRES"]], 0.0)
            his_rmw = setcd__safe_float(his_row[self.col2idx["USA_RMW"]], 0.0)
            # All three values enter historical ratios; reject invalid denominators.
            if not all(setcd__value_exists(v) for v in (his_msw, his_mslp, his_rmw)):
                continue

            his_r34ne = setcd__safe_float(his_row[self.col2idx["USA_R34_NE"]], 0.0)
            his_r34se = setcd__safe_float(his_row[self.col2idx["USA_R34_SE"]], 0.0)
            his_r34sw = setcd__safe_float(his_row[self.col2idx["USA_R34_SW"]], 0.0)
            his_r34nw = setcd__safe_float(his_row[self.col2idx["USA_R34_NW"]], 0.0)

            his_r50ne = setcd__safe_float(his_row[self.col2idx["USA_R50_NE"]], 0.0)
            his_r50se = setcd__safe_float(his_row[self.col2idx["USA_R50_SE"]], 0.0)
            his_r50sw = setcd__safe_float(his_row[self.col2idx["USA_R50_SW"]], 0.0)
            his_r50nw = setcd__safe_float(his_row[self.col2idx["USA_R50_NW"]], 0.0)

            if self.idx_r34 is not None:
                his_r34 = setcd__safe_float(his_row[self.idx_r34], 0.0)
            else:
                his_r34 = setcd__safe_mean_of_valid([his_r34ne, his_r34se, his_r34sw, his_r34nw])

            if self.idx_r50 is not None:
                his_r50 = setcd__safe_float(his_row[self.idx_r50], 0.0)
            else:
                his_r50 = setcd__safe_mean_of_valid([his_r50ne, his_r50se, his_r50sw, his_r50nw])

            if self.idx_level is not None:
                his_level = int(setcd__safe_float(his_row[self.idx_level], setcd__msw_to_level(his_msw)))
            else:
                his_level = setcd__msw_to_level(his_msw)

            if not all(setcd__value_exists(v) for v in (his_r34, his_r50)):
                continue
            if his_rmw > his_r34:
                continue

            his_tcf = 1 - (his_rmw / his_r34)
            his_isr = his_msw / his_rmw
            his_pwr = his_mslp / his_msw
            his_prr = his_mslp / his_r34
            his_rrr = 1 - (his_r50 / his_r34)

            self.sample_keys.append(labels_dic_key)
            self.pre_tcfs.append(his_tcf)
            self.pre_isrs.append(his_isr)
            self.pre_pwrs.append(his_pwr)
            self.pre_prrs.append(his_prr)
            self.pre_rrrs.append(his_rrr)
            self.plevels.append(his_level)
            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.levels.append(level)
            self.norm_ts.append(t)

            self.pxh_k8_btemps.append(os.path.join(k8_path, pxh_k8_fname))
            self.k8_btemps.append(os.path.join(k8_path, filename))

            self.masks.append(mask)
            self.msws.append(msw)
            self.mslps.append(mslp)
            self.rmws.append(rmw)
            self.r34nes.append(r34ne)
            self.r34ses.append(r34se)
            self.r34sws.append(r34sw)
            self.r34nws.append(r34nw)
            self.r50nes.append(r50ne)
            self.r50ses.append(r50se)
            self.r50sws.append(r50sw)
            self.r50nws.append(r50nw)
            self.r64nes.append(r64ne)
            self.r64ses.append(r64se)
            self.r64sws.append(r64sw)
            self.r64nws.append(r64nw)

        if len(self.msws) == 0:
            raise RuntimeError(
                "No usable samples; check paths, label rows, columns, storm ages and invalid-sample records."
            )

        print("msws max = {}, min = {}".format(max(self.msws), min(self.msws)))
        print("mslps max = {}, min = {}".format(max(self.mslps), min(self.mslps)))
        print("rmws max = {}, min = {}".format(max(self.rmws), setcd__safe_min_nonzero(self.rmws)))
        print("r34nes max = {}, min = {}".format(max(self.r34nes), setcd__safe_min_nonzero(self.r34nes)))
        print("r34ses max = {}, min = {}".format(max(self.r34ses), setcd__safe_min_nonzero(self.r34ses)))
        print("r34sws max = {}, min = {}".format(max(self.r34sws), setcd__safe_min_nonzero(self.r34sws)))
        print("r34nws max = {}, min = {}".format(max(self.r34nws), setcd__safe_min_nonzero(self.r34nws)))
        print("r50nes max = {}, min = {}".format(max(self.r50nes), setcd__safe_min_nonzero(self.r50nes)))
        print("r50ses max = {}, min = {}".format(max(self.r50ses), setcd__safe_min_nonzero(self.r50ses)))
        print("r50sws max = {}, min = {}".format(max(self.r50sws), setcd__safe_min_nonzero(self.r50sws)))
        print("r50nws max = {}, min = {}".format(max(self.r50nws), setcd__safe_min_nonzero(self.r50nws)))
        print("r64nes max = {}, min = {}".format(max(self.r64nes), setcd__safe_min_nonzero(self.r64nes)))
        print("r64ses max = {}, min = {}".format(max(self.r64ses), setcd__safe_min_nonzero(self.r64ses)))
        print("r64sws max = {}, min = {}".format(max(self.r64sws), setcd__safe_min_nonzero(self.r64sws)))
        print("r64nws max = {}, min = {}".format(max(self.r64nws), setcd__safe_min_nonzero(self.r64nws)))
        print("t max = {}, min = {}".format(max(self.ts), min(self.ts)))
        print("pre_tcf max = {}, min = {}".format(max(self.pre_tcfs), min(self.pre_tcfs)))
        print("pre_isr max = {}, min = {}".format(max(self.pre_isrs), min(self.pre_isrs)))
        print("pre_pwr max = {}, min = {}".format(max(self.pre_pwrs), min(self.pre_pwrs)))
        print("pre_prr max = {}, min = {}".format(max(self.pre_prrs), min(self.pre_prrs)))
        print("pre_rrr max = {}, min = {}".format(max(self.pre_rrrs), min(self.pre_rrrs)))

        for i in range(len(self.msws)):
            self.msws[i] = (self.msws[i] - 35.00) / (185.00 - 35.00)
            self.mslps[i] = (self.mslps[i] - 872.00) / (1011.00 - 872.00)
            self.rmws[i] = (self.rmws[i] - 5.00) / (300.00 - 5.00)

            self.r34nes[i] = (self.r34nes[i] - 5.00) / (780.00 - 5.00)
            self.r34ses[i] = (self.r34ses[i] - 5.00) / (660.00 - 5.00)
            self.r34sws[i] = (self.r34sws[i] - 5.00) / (660.00 - 5.00)
            self.r34nws[i] = (self.r34nws[i] - 5.00) / (600.00 - 5.00)

            self.r50nes[i] = (self.r50nes[i] - 5.00) / (300.00 - 5.00)
            self.r50ses[i] = (self.r50ses[i] - 5.00) / (270.00 - 5.00)
            self.r50sws[i] = (self.r50sws[i] - 5.00) / (330.00 - 5.00)
            self.r50nws[i] = (self.r50nws[i] - 10.00) / (360.00 - 10.00)

            self.r64nes[i] = (self.r64nes[i] - 5.00) / (180.00 - 5.00)
            self.r64ses[i] = (self.r64ses[i] - 5.00) / (180.00 - 5.00)
            self.r64sws[i] = (self.r64sws[i] - 5.00) / (180.00 - 5.00)
            self.r64nws[i] = (self.r64nws[i] - 5.00) / (300.00 - 5.00)

            self.norm_ts[i] = (self.ts[i] - 12.00) / (606.00 - 12.00)
            self.pre_isrs[i] = (self.pre_isrs[i] - 0.33) / (37.00 - 0.33)
            self.pre_pwrs[i] = (self.pre_pwrs[i] - 4.71) / (20.18 - 4.71)
            self.pre_prrs[i] = (self.pre_prrs[i] - 1.52) / (99.80 - 1.52)
            self.pre_rrrs[i] = (self.pre_rrrs[i] - (-0.36)) / (0.92 - (-0.36))

    def __len__(self):
        return len(self.k8_btemps)

    def __getitem__(self, index):
        btemp_file_path = self.k8_btemps[index]
        k8_btemp = setcd_default_loader(btemp_file_path)

        pxh_btemp_file_path = self.pxh_k8_btemps[index]
        pxh_k8_btemp = setcd_default_loader(pxh_btemp_file_path)

        msw = self.msws[index]
        mslp = self.mslps[index]
        rmw = self.rmws[index]
        r34ne = self.r34nes[index]
        r34se = self.r34ses[index]
        r34sw = self.r34sws[index]
        r34nw = self.r34nws[index]
        r50ne = self.r50nes[index]
        r50se = self.r50ses[index]
        r50sw = self.r50sws[index]
        r50nw = self.r50nws[index]
        r64ne = self.r64nes[index]
        r64se = self.r64ses[index]
        r64sw = self.r64sws[index]
        r64nw = self.r64nws[index]

        group1_label = torch.tensor([msw, mslp, rmw], dtype=torch.float32).reshape(1, 3)
        group2_label = torch.tensor([
            r34ne, r50ne, r64ne,
            r34se, r50se, r64se,
            r34sw, r50sw, r64sw,
            r34nw, r50nw, r64nw
        ], dtype=torch.float32).reshape(4, 3)
        group2_label_mask = self.masks[index].reshape(4, 3)

        level = self.levels[index]
        lat = self.lats[index]
        lon = self.lons[index]
        pre_tcf = round(self.pre_tcfs[index], 3)
        pre_isr = round(self.pre_isrs[index], 3)
        pre_pwr = round(self.pre_pwrs[index], 3)
        pre_prr = round(self.pre_prrs[index], 3)
        pre_rrr = round(self.pre_rrrs[index], 3)
        t = self.norm_ts[index]
        pre_level = self.plevels[index]

        pr_list = [pre_tcf, pre_isr, pre_pwr, pre_prr, pre_rrr]
        pr = torch.tensor(pr_list, dtype=torch.float32)

        plt_list = [pre_level, t]
        plt = torch.tensor(plt_list, dtype=torch.float32)

        sample_key = self.sample_keys[index]

        sample = {
            'sample_key': sample_key,
            'btemp': k8_btemp,
            'pxh_btemp': pxh_k8_btemp,
            'k8_btemp': k8_btemp,
            'pxh_k8_btemp': pxh_k8_btemp,
            'group1_label': group1_label,
            'group2_label': group2_label,
            'group2_label_mask': group2_label_mask,
            'level': level,
            'lat': torch.tensor(lat, dtype=torch.float32),
            'lon': torch.tensor(lon, dtype=torch.float32),
            'pr': pr,
            'plt': plt
        }
        return sample



def create_dataset(path):
    """Select the paper-specific loader using Config.dataset_name."""
    cls = PDTCDataset if config.dataset_name == 'PDTC' else SETCDSatDataset
    return cls(path)
