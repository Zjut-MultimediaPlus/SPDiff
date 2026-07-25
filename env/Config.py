# train
epochs = 200
batch_size = 48
device = 'cuda:0'  # cpu or 'cuda:0'
id_dim = 16

denoise_sample = 5
mc_sample = 4
timesteps = 100

group_num = 2
ingroup_tuple = (3, 12)

group1_size = 1
group1_dim = 3
group2_size = 4
group2_dim = 3

task_num = 6

group_size = 4
tn_size = 15
sample = 12

''' pdtc dataset '''
labels_path = "/opt/data/private/norm_data_npy/Full2015_2023_Dataets/AllRadi_Labels_2015_2024.pkl"
k8_sta_path = "/opt/data/private/norm_data_npy/Full2015_2023_Dataets/newk8_norm" \
              "" \
              "_values_split.pkl"
p12hpr_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/pre12h_labels.pkl'
train_k8_path = '/TCdata/k89_4ch1/train/'
valid_k8_path = '/TCdata/k89_4ch1/valid/'
predict_k8_path = '/TCdata/k89_4ch1/test/'

model_output_dir = '/opt/data/private/model/ProbDiffusion/Backbone/'
predict_model = '/opt/data/private/model/ProbDiffusion/Backbone/epoch_200.pth'
save_fig_dir = '/opt/data/private/model/ProbDiffusion/Backbone/exp_img/'

save_cep_dir = '/opt/data/private/model/ProbDiffusion/ConceptBank/concept_embs_clip.pt'

num_workers = 4
best_loss = 0.005
save_model_iter = 20

data_format = 'npy'
