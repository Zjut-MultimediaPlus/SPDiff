import clip
import numpy as np
import torch
import os
from tqdm import tqdm
import env.Config as config
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

def find_previous_hours(time_str, x):
    # 解析时间字符串
    time_format = "%Y%m%d%H"
    time = datetime.strptime(time_str, time_format)

    # 计算前x小时
    previous_time = time - timedelta(hours=x)

    # 将结果转换为字符串
    previous_time_str = previous_time.strftime(time_format)

    return previous_time_str

def get_previous_npy_index(npy_index, x):
    """
    根据给定的 npy_index 和小时数 x，返回前 x 小时的 npy_index。
    npy_index 格式: '年份_台风名字_时间'，例如 '2019_台风名字_2019010506'
    """
    # 解析 npy_index
    parts = npy_index.split('_')
    year_typhoon_name = parts[0] + '_' + parts[1]
    time_str = parts[2]  # 例如 '2019010506'

    # 将时间字符串转换为 datetime 对象
    time_format = '%Y%m%d%H'
    time_obj = datetime.strptime(time_str, time_format)

    # 减去 x 小时
    previous_time_obj = time_obj - timedelta(hours=x)

    # 将 datetime 对象转换回字符串
    previous_time_str = previous_time_obj.strftime(time_format)

    # 重新拼接生成新的 npy_index
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

'''full整体归一'''
def get_fnorm_now_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        '''处理负值（按通道处理，用每个通道的均值填充）'''
        # channel_mean = torch.mean(x[c][x[c] > 0])
        # x_norm[c] = torch.where(x[c] < 0, 0, x[c])
        '''train'''
        maxv = statistic_dic['now_train'][c][0]
        minv = statistic_dic['now_train'][c][1]
        '''all'''
        # maxv = statistic_dic['now_all'][c][0]
        # minv = statistic_dic['now_all'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)

    return x_norm

def get_fnorm_p3h_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        '''处理负值（按通道处理，用每个通道的均值填充）'''
        # channel_mean = torch.mean(x[c][x[c] > 0])
        # x_norm[c] = torch.where(x[c] <= 0, 0, x[c])
        '''train'''
        maxv = statistic_dic['p3h_train'][c][0]
        minv = statistic_dic['p3h_train'][c][1]
        '''all'''
        # maxv = statistic_dic['p3h_all'][c][0]
        # minv = statistic_dic['p3h_all'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)
    return x_norm

class TCDataset_HisPR():
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

        self.labels_dic = load_dict_from_pickle(config.labels_path)

        # 获取目录下的所有文件
        k8_files = os.listdir(k8_path)
        k8_files_set = set(k8_files)

        self.k8_sta_dic = load_dict_from_pickle(config.k8_sta_path)
        pre12h_labels_data = load_dict_from_pickle(config.p12hpr_pth)
        self.level_name = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']
        # 遍历文件
        for i, filename in enumerate(k8_files):
            fname_split = filename.split("_")
            tcname = fname_split[1]
            year = fname_split[0]
            '''e.g. 2023_BOLAVEN_2023100809'''
            if len(fname_split) == 3:
                isotime = fname_split[2][:-4]
            else:
                isotime = fname_split[2]
            labels_dic_key = year + "_" + tcname + "_" + isotime

            '''e.g. BOLAVEN_2023100806'''
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

            iso_time, lat, lon, t, level, mslp, msw, rmw, r34, r50, r34ne, r34se, r34sw, r34nw, r50ne, r50se, r50sw, r50nw, r64ne, r64se, r64sw, r64nw = self.labels_dic[labels_dic_key]

            # labels = [msw, mslp, rmw,
            #           r34ne, r34se, r34sw, r34nw,
            #           r50ne, r50se, r50sw, r50nw,
            #           r64ne, r64se, r64sw, r64nw]
            labels = [msw, mslp, rmw,
                      r34ne, r50ne, r64ne,
             r34se, r50se, r64se,
             r34sw, r50sw, r64sw,
             r34nw, r50nw, r64nw]
            mask = [int(i != 0) for i in labels]
            mask = torch.tensor(mask)

            his_dic_index = get_previous_npy_index(labels_dic_key, 12)
            if his_dic_index not in self.labels_dic.keys():
                continue
            his_level, his_mslp, his_msw, his_rmw, his_r34, his_r50, \
            his_r34ne, his_r34se, his_r34sw, his_r34nw, \
            his_r50ne, his_r50se, his_r50sw, his_r50nw, \
            his_r64ne, his_r64se, his_r64sw, his_r64nw = \
                self.labels_dic[his_dic_index][4:]
            if his_r34 == 0 or his_rmw == 0 or his_r50 == 0:
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
            # self.plevels.append(level_name[his_level + 1])
            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.levels.append(level)
            self.norm_ts.append(t)

            self.pxh_k8_btemps.append(k8_path + pxh_k8_fname)
            self.k8_btemps.append(k8_path + filename)

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
        # print("lat max = {}, min = {}".format(max(self.lats), min(self.lats)))
        # print("lon max = {}, min = {}".format(max(self.lons), min(self.lons)))
        print("t max = {}, min = {}".format(max(self.ts), min(self.ts)))
        print("pre_tcf max = {}, min = {}".format(max(self.pre_tcfs), min(self.pre_tcfs)))
        print("pre_isr max = {}, min = {}".format(max(self.pre_isrs), min(self.pre_isrs)))
        print("pre_pwr max = {}, min = {}".format(max(self.pre_pwrs), min(self.pre_pwrs)))
        print("pre_prr max = {}, min = {}".format(max(self.pre_prrs), min(self.pre_prrs)))
        print("pre_rrr max = {}, min = {}".format(max(self.pre_rrrs), min(self.pre_rrrs)))

        # 标签归一化
        for i in range(len(self.msws)):
            '''pre 12 h PR'''
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
            # self.lats[i] = (self.lats[i] - (-32.038)) / (42.491 - (-32.038))
            # self.lons[i] = (self.lons[i] - 196.1) / (196.1 - 83.892)
            # self.norm_ts[i] = (self.ts[i] - 12) / (459 - 12)
            self.pre_isrs[i] = (self.pre_isrs[i] - 0.38) / (34 - 0.38)
            self.pre_pwrs[i] = (self.pre_pwrs[i] - (5.18)) / (20.02 - (5.18))
            self.pre_prrs[i] = (self.pre_prrs[i] - (2.76)) / (60.99 - (2.76))

    def __len__(self):
        return len(self.k8_btemps)

    def __getitem__(self, index):
        # 4, 156, 156
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
        # label = [msw, mslp, rmw, r34ne, r34se, r34sw, r34nw, r50ne, r50se, r50sw, r50nw, r64ne, r64se, r64sw, r64nw]
        label = [msw, mslp, rmw,
                  r34ne, r50ne, r64ne,
                  r34se, r50se, r64se,
                  r34sw, r50sw, r64sw,
                  r34nw, r50nw, r64nw]
        label_mask = self.masks[index]

        lat = self.lats[index]
        lon = self.lons[index]
        pre_tcf = round(self.pre_tcfs[index], 3)
        pre_isr = round(self.pre_isrs[index], 3)
        pre_pwr = round(self.pre_pwrs[index], 3)
        pre_prr = round(self.pre_prrs[index], 3)
        pre_rrr = round(self.pre_rrrs[index], 3)
        t = self.ts[index]
        pre_level = self.plevels[index]
        pr_list = [pre_tcf, pre_isr, pre_pwr, pre_prr, pre_rrr]
        pr = torch.tensor(pr_list)
        sample = {'tca_label': torch.tensor(label),
                   'label_mask': label_mask,
                   'k8_btemp': k8_btemp,
                  'pxh_k8_btemp': pxh_k8_btemp, 'pr': pr
                  }
        return sample

class TCDataset_HisPR_QuadTG():
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

        self.labels_dic = load_dict_from_pickle(config.labels_path)

        # 获取目录下的所有文件
        k8_files = os.listdir(k8_path)
        k8_files_set = set(k8_files)

        self.k8_sta_dic = load_dict_from_pickle(config.k8_sta_path)
        pre12h_labels_data = load_dict_from_pickle(config.p12hpr_pth)
        self.level_name = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']
        # 遍历文件
        for i, filename in enumerate(k8_files):
            fname_split = filename.split("_")
            tcname = fname_split[1]
            year = fname_split[0]
            '''e.g. 2023_BOLAVEN_2023100809'''
            if len(fname_split) == 3:
                isotime = fname_split[2][:-4]
            else:
                isotime = fname_split[2]
            labels_dic_key = year + "_" + tcname + "_" + isotime

            '''e.g. BOLAVEN_2023100806'''
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

            iso_time, lat, lon, t, level, mslp, msw, rmw, r34, r50, r34ne, r34se, r34sw, r34nw, r50ne, r50se, r50sw, r50nw, r64ne, r64se, r64sw, r64nw = self.labels_dic[labels_dic_key]

            labels = [msw, mslp, rmw,
                      r34ne, r50ne, r64ne,
                      r34se, r50se, r64se,
                      r34sw, r50sw, r64sw,
                      r34nw, r50nw, r64nw]
            mask = [int(i != 0) for i in labels]
            mask = torch.tensor(mask)

            his_dic_index = get_previous_npy_index(labels_dic_key, 12)
            if his_dic_index not in self.labels_dic.keys():
                continue
            his_level, his_mslp, his_msw, his_rmw, his_r34, his_r50, \
            his_r34ne, his_r34se, his_r34sw, his_r34nw, \
            his_r50ne, his_r50se, his_r50sw, his_r50nw, \
            his_r64ne, his_r64se, his_r64sw, his_r64nw = \
                self.labels_dic[his_dic_index][4:]
            if his_r34 == 0 or his_rmw == 0 or his_r50 == 0:
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
            # self.plevels.append(level_name[his_level + 1])
            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.levels.append(level)
            self.norm_ts.append(t)

            self.pxh_k8_btemps.append(k8_path + pxh_k8_fname)
            self.k8_btemps.append(k8_path + filename)

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
        # print("lat max = {}, min = {}".format(max(self.lats), min(self.lats)))
        # print("lon max = {}, min = {}".format(max(self.lons), min(self.lons)))
        print("t max = {}, min = {}".format(max(self.ts), min(self.ts)))
        print("pre_tcf max = {}, min = {}".format(max(self.pre_tcfs), min(self.pre_tcfs)))
        print("pre_isr max = {}, min = {}".format(max(self.pre_isrs), min(self.pre_isrs)))
        print("pre_pwr max = {}, min = {}".format(max(self.pre_pwrs), min(self.pre_pwrs)))
        print("pre_prr max = {}, min = {}".format(max(self.pre_prrs), min(self.pre_prrs)))
        print("pre_rrr max = {}, min = {}".format(max(self.pre_rrrs), min(self.pre_rrrs)))

        # 标签归一化
        for i in range(len(self.msws)):
            '''pre 12 h PR'''
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
            # self.lats[i] = (self.lats[i] - (-32.038)) / (42.491 - (-32.038))
            # self.lons[i] = (self.lons[i] - 196.1) / (196.1 - 83.892)
            # self.norm_ts[i] = (self.ts[i] - 12) / (459 - 12)
            self.pre_isrs[i] = (self.pre_isrs[i] - 0.38) / (34 - 0.38)
            self.pre_pwrs[i] = (self.pre_pwrs[i] - (5.18)) / (20.02 - (5.18))
            self.pre_prrs[i] = (self.pre_prrs[i] - (2.76)) / (60.99 - (2.76))

    def __len__(self):
        return len(self.k8_btemps)

    def __getitem__(self, index):
        # 4, 156, 156
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
        '''labels = [msw, mslp, rmw, 
                      r34ne, r50ne, r64ne, 
                      r34se, r50se, r64se, 
                      r34sw, r50sw, r64sw,
                      r34nw, r50nw, r64nw]
        '''
        group_label = torch.tensor([msw, mslp, rmw,
                      r34ne, r50ne, r64ne,
                      r34se, r50se, r64se,
                      r34sw, r50sw, r64sw,
                      r34nw, r50nw, r64nw])
        group_label_mask = self.masks[index]

        group_label = group_label.reshape(5, 3)
        group_label_mask = group_label_mask.reshape(5, 3)

        lat = self.lats[index]
        lon = self.lons[index]
        pre_tcf = round(self.pre_tcfs[index], 3)
        pre_isr = round(self.pre_isrs[index], 3)
        pre_pwr = round(self.pre_pwrs[index], 3)
        pre_prr = round(self.pre_prrs[index], 3)
        pre_rrr = round(self.pre_rrrs[index], 3)
        t = self.ts[index]
        pre_level = self.plevels[index]
        pr_list = [pre_tcf, pre_isr, pre_pwr, pre_prr, pre_rrr]
        pr = torch.tensor(pr_list)
        sample = {'group_label': group_label,
                   'group_label_mask': group_label_mask,
                   'k8_btemp': k8_btemp,
                  'pxh_k8_btemp': pxh_k8_btemp, 'pr': pr
                  }
        return sample

def classify_quadrant_pattern(
    radii,
    tol_uniform_ratio=0.2,
    single_ratio=1.4,
    delta_dual=0.25,
    min_radius=0,
):
    """
    radii: 长度 4 的 array-like, 顺序为 [NE, SE, SW, NW]
    返回:
        'Quadrants Uniform' / 'Single Quadrant Expanded' /
        'Dual Quadrants Extended' / 'Irregular Expansion'
    """
    r = np.asarray(radii, dtype=float)

    # 过滤极小/无效值
    valid_mask = r >= min_radius
    if valid_mask.sum() < 2:
        return "Irregular Expansion"

    r_valid = r[valid_mask]
    max_r = r_valid.max()
    min_r = r_valid.min()
    if max_r <= 0:
        return "Irregular Expansion"

    # 1) 均匀性：相对极差
    spread = (max_r - min_r) / max_r
    if spread < tol_uniform_ratio:
        return "Quadrants Uniform"

    # 2) 单象限特别大：最大值显著大于第二大值
    full_idx = np.arange(4)
    valid_idx = full_idx[valid_mask]
    sort_order = np.argsort(-r_valid)  # 从大到小
    idx_max = valid_idx[sort_order[0]]
    max_r = r[idx_max]
    if len(r_valid) >= 2:
        second_r = r_valid[sort_order[1]]
    else:
        second_r = min_r

    if second_r > 0 and (max_r / second_r) >= single_ratio:
        return "Single Quadrant Expanded"

    # 3) 两个象限特别大：超过 mean*(1+delta_dual)，且相邻或对称
    mean_r = r_valid.mean()
    big_mask = r >= mean_r * (1.0 + delta_dual)
    big_indices = np.where(big_mask)[0]

    # 邻接 + 对称象限对
    adjacent_pairs = {(0, 1), (1, 2), (2, 3), (0, 3)}  # NE-SE, SE-SW, SW-NW, NE-NW
    opposite_pairs = {(0, 2), (1, 3)}                  # NE-SW, SE-NW

    if len(big_indices) == 2:
        pair = tuple(sorted(big_indices.tolist()))
        if pair in adjacent_pairs or pair in opposite_pairs:
            return "Dual Quadrants Extended"

    # 4) 其余情况认为是不规则
    return "Irregular Expansion"

class TCDataset_HisPR_TwoTG():
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

        # RIFlag
        self.ri_flags = []

        # 新增：SymFlags（按 R34/R50/R64 三个等级）
        self.sym_flags_r34 = []
        self.sym_flags_r50 = []
        self.sym_flags_r64 = []

        self.labels_dic = load_dict_from_pickle(config.labels_path)

        k8_files = os.listdir(k8_path)
        k8_files_set = set(k8_files)

        self.k8_sta_dic = load_dict_from_pickle(config.k8_sta_path)
        pre12h_labels_data = load_dict_from_pickle(config.p12hpr_pth)
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
                if fname_split[3] in ['rotate45.npy', 'rotate90.npy', 'rotate135.npy', 'rotate270.npy']:
                    continue

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

            if his_r34 == 0 or his_rmw == 0 or his_r50 == 0:
                continue
            if his_rmw > his_r34:
                continue

            his_tcf = 1 - (his_rmw / his_r34)
            his_isr = his_msw / his_rmw
            his_pwr = his_mslp / his_msw
            his_prr = his_mslp / his_r34
            his_rrr = 1 - (his_r50 / his_r34)

            # RIFlag
            ri_flag = 0
            prev_key = get_previous_npy_index(labels_dic_key, 24)
            if prev_key in self.labels_dic.keys():
                prev_msw = self.labels_dic[prev_key][6]
                if (msw - prev_msw) >= 30:
                    ri_flag = 1

            # -----------------------------
            # 新增：SymFlags
            # 对每个风圈等级按四象限 [NE, SE, SW, NW] 做模式分类
            # 只有 "Quadrants Uniform" 记为 1，其余三类记为 0
            # -----------------------------
            pat_r34 = classify_quadrant_pattern([r34ne, r34se, r34sw, r34nw])
            pat_r50 = classify_quadrant_pattern([r50ne, r50se, r50sw, r50nw])
            pat_r64 = classify_quadrant_pattern([r64ne, r64se, r64sw, r64nw])

            sym_flag_r34 = 1 if pat_r34 == "Quadrants Uniform" else 0
            sym_flag_r50 = 1 if pat_r50 == "Quadrants Uniform" else 0
            sym_flag_r64 = 1 if pat_r64 == "Quadrants Uniform" else 0

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

            self.pxh_k8_btemps.append(k8_path + pxh_k8_fname)
            self.k8_btemps.append(k8_path + filename)

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

            self.ri_flags.append(ri_flag)

            self.sym_flags_r34.append(sym_flag_r34)
            self.sym_flags_r50.append(sym_flag_r50)
            self.sym_flags_r64.append(sym_flag_r64)

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
        print("RI sample count = {}, non-RI sample count = {}".format(
            sum(self.ri_flags), len(self.ri_flags) - sum(self.ri_flags)
        ))
        print("R34 SymFlag=1 count = {}, SymFlag=0 count = {}".format(
            sum(self.sym_flags_r34), len(self.sym_flags_r34) - sum(self.sym_flags_r34)
        ))
        print("R50 SymFlag=1 count = {}, SymFlag=0 count = {}".format(
            sum(self.sym_flags_r50), len(self.sym_flags_r50) - sum(self.sym_flags_r50)
        ))
        print("R64 SymFlag=1 count = {}, SymFlag=0 count = {}".format(
            sum(self.sym_flags_r64), len(self.sym_flags_r64) - sum(self.sym_flags_r64)
        ))

        # 标签归一化
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
        pre_level = self.plevels[index]
        pr_list = [pre_tcf, pre_isr, pre_pwr, pre_prr, pre_rrr]
        pr = torch.tensor(pr_list, dtype=torch.float32)
        sample_key = self.sample_keys[index]
        ri_flag = self.ri_flags[index]

        sym_flags = torch.tensor([
            self.sym_flags_r34[index],
            self.sym_flags_r50[index],
            self.sym_flags_r64[index]
        ], dtype=torch.long)

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
            'RIFlag': torch.tensor(ri_flag, dtype=torch.long),

            # 新增
            'SymFlags': sym_flags,                  # shape=(3,), [R34, R50, R64]
            'SymFlag_R34': torch.tensor(self.sym_flags_r34[index], dtype=torch.long),
            'SymFlag_R50': torch.tensor(self.sym_flags_r50[index], dtype=torch.long),
            'SymFlag_R64': torch.tensor(self.sym_flags_r64[index], dtype=torch.long),
        }
        return sample




