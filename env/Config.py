"""Shared PDTC / SETCD-Sat experiment configuration."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Option 1: edit the defaults below (dataset: PDTC or SETCD-Sat; root: server resource folder).
# Option 2: set SPDIFF_DATASET and SPDIFF_DATA_ROOT; environment values override the defaults.
dataset_name = os.environ.get('SPDIFF_DATASET', 'PDTC')
DATA_ROOT = Path(os.environ.get('SPDIFF_DATA_ROOT', REPO_ROOT / 'downloads' / 'SPDiff-Data-Resource')).expanduser().resolve()
if dataset_name not in ('PDTC', 'SETCD-Sat'):
    raise ValueError('dataset_name must be PDTC or SETCD-Sat')
dataset_root = DATA_ROOT / dataset_name
metadata_root = dataset_root / 'Metadata'
samples_root = dataset_root / 'Satellite_data' / ('k89_4ch1' if dataset_name == 'PDTC' else 'Gridsat_IR')
train_data_path = str(samples_root / 'train')
valid_data_path = str(samples_root / 'valid')
test_data_path = str(samples_root / 'test')
labels_path = str(metadata_root / 'AllRadii_Labels.pkl')
k8_sta_path = str(metadata_root / 'satellite_ir_norm_values_split.pkl')
ibtracs_row_dict_path = str(metadata_root / 'ibtracs_row_dict.pkl')
ibtracs_columns_path = str(metadata_root / 'ibtracs_columns.pkl')
nan_over_half_first_channel_path = str(metadata_root / 'nan_over_half_first_channel.pkl')
ibtracs_age_hours_path = str(metadata_root / 'ibtracs_age_hours.pkl')
save_cep_dir = os.environ.get('SPDIFF_CONCEPT_EMBS', str(DATA_ROOT / 'concept_embs_clip.pt'))
image_channels = 4 if dataset_name == 'PDTC' else 1
epochs = int(os.environ.get('SPDIFF_EPOCHS', 200))
batch_size = int(os.environ.get('SPDIFF_BATCH_SIZE', 48))
device = os.environ.get('SPDIFF_DEVICE', 'cuda:0')
num_workers = int(os.environ.get('SPDIFF_NUM_WORKERS', 4))
# Optional validation uses forward losses, not slow diffusion sampling.
use_validation = os.environ.get('SPDIFF_USE_VALIDATION', '0') == '1'
validation_interval = int(os.environ.get('SPDIFF_VALIDATION_INTERVAL', 10))
# Zero means all validation batches; positive values limit validation cost.
validation_max_batches = int(os.environ.get('SPDIFF_VALIDATION_MAX_BATCHES', 0))
if validation_interval < 1 or validation_max_batches < 0:
    raise ValueError('Validation interval must be positive and max batches nonnegative.')
model_output_dir = os.environ.get('SPDIFF_OUTPUT_DIR', str(REPO_ROOT / 'runs' / dataset_name))
predict_model = os.environ.get('SPDIFF_CHECKPOINT', str(Path(model_output_dir) / f'epoch_{epochs}.pth'))
save_fig_dir = str(Path(model_output_dir) / 'figs')
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
best_loss = 0.005
save_model_iter = 20
data_format = 'npy'

def validate_paths(training=True, split=None):
    metadata = ([labels_path, k8_sta_path] if dataset_name == 'PDTC' else
                [ibtracs_row_dict_path, ibtracs_columns_path, ibtracs_age_hours_path,
                 nan_over_half_first_channel_path])
    required = metadata + [save_cep_dir]
    if split is None and not training:
        required.append(predict_model)
    missing = [p for p in required if not Path(p).is_file()]
    if split is not None:
        # Data inspection requires only the chosen split, not a model checkpoint.
        if split not in ('train', 'valid', 'test'):
            raise ValueError('split must be train, valid or test')
        splits = [str(samples_root / split)]
    else:
        splits = ([train_data_path] + ([valid_data_path] if use_validation else [])) if training else [test_data_path]
    missing += [p for p in splits if not Path(p).is_dir() or not any(Path(p).glob('*.npy'))]
    if missing:
        raise FileNotFoundError('Missing resources or empty splits:\n' + '\n'.join(missing))
