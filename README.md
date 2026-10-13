# SPDiff

Code for **SPDiff: Semantic Proposal-guided Diffusion for Reliable Asymmetric Tropical Cyclone Estimation**, under review at KDD AI4S 2027. The repository provides dataset access, usage instructions, and code for end-to-end training and evaluation.

SPDiff estimates maximum sustained wind, minimum sea-level pressure, radius of
maximum wind, and quadrant-wise wind radii at 34, 50 and 64 knots. PDTC and SETCD-Sat
share the same configuration, dataset entry point and model implementation.

## Installation

Use a Python environment with compatible PyTorch and torchvision versions for
your hardware. Install the runtime dependencies:

```bash
python -m pip install torch torchvision numpy matplotlib tqdm einops
```

## Data preparation

Follow [DATA_README.md](DATA_README.md) to download and extract the resources. Set `SPDIFF_DATA_ROOT` to the resource folder on your server. Both datasets use the shared `concept_embs_clip.pt` located at that root.

## Train and evaluate

Run commands from the repository root. Dataset selection uses `PDTC` or `SETCD-Sat`; you can also edit `dataset_name` and `DATA_ROOT` in `env/Config.py`.

### PDTC

```bash
export SPDIFF_DATA_ROOT=/path/to/SPDiff-Data-Resource
export SPDIFF_DATASET=PDTC
export SPDIFF_OUTPUT_DIR="$PWD/runs/PDTC"
python scripts/check_data.py --split train --load-sample
python Train_TwoTGCD_Conf.py
export SPDIFF_CHECKPOINT="$SPDIFF_OUTPUT_DIR/epoch_300.pth"
python Test_TwoTGCD_Conf.py
```

### SETCD-Sat

```bash
export SPDIFF_DATA_ROOT=/path/to/SPDiff-Data-Resource
export SPDIFF_DATASET=SETCD-Sat
export SPDIFF_OUTPUT_DIR="$PWD/runs/SETCD-Sat"
python scripts/check_data.py --split train --load-sample
python Train_TwoTGCD_Conf.py
export SPDIFF_CHECKPOINT="$SPDIFF_OUTPUT_DIR/epoch_300.pth"
python Test_TwoTGCD_Conf.py
```

The default settings are 300 epochs, a batch size of 48, four data workers, random seed 0, and device `cuda:0`. These settings can be overridden in `Config.py` or through the environment variables `SPDIFF_EPOCHS`, `SPDIFF_BATCH_SIZE`, `SPDIFF_NUM_WORKERS`, and `SPDIFF_DEVICE`. The concept-bank file can also be specified using `SPDIFF_CONCEPT_EMBS`.

Evaluation reports MAE after dataset-specific inverse normalization, including per-quadrant errors and errors of the mean radii across valid quadrants. During training, periodic checkpoints and a final checkpoint (`epoch_<epochs>.pth`) are saved to the output directory.

## Project Structure

```text
env/
├── Config.py                  # Dataset paths and experiment configurations
└── Dataset.py                 # Dataset loaders with automatic dataset selection
utils/
├── Condition_Encoder.py       # Shared encoder with dataset-specific input
├── TwoDenoiseModel.py         # Shared denoising model
└── Diffusion.py               # Diffusion framework and concept-bank loading
scripts/
└── check_data.py              # Data validation and optional sample loading
TrainNetworkHelper_TwoTG.py    # Training utilities
Train_TwoTGCD_Conf.py          # Unified training entry point
Test_TwoTGCD_Conf.py           # Unified evaluation entry point
DATA_README.md                 # Dataset download instructions and directory
README.md
```
