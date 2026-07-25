# SPDiff

This repository contains the code for the paper **"SPDiff: Semantic Proposal-guided Diffusion for Reliable Asymmetric Tropical Cyclone Estimation,"** currently under review at KDD AI4S 2027.

SPDiff is a conditional diffusion model for tropical cyclone intensity and asymmetric wind-radii estimation. The model uses multi-modal TC data to estimate:

- Maximum sustained wind (`msw`)
- Minimum sea-level pressure (`mslp`)
- Radius of maximum wind (`rmw`)
- Quadrant wind radii for 34-, 50-, and 64-knot winds

## Project Structure

```text
SPDiff_Code/
├── env/
│   ├── Config.py                  # Paths and experiment settings
│   ├── Dataset.py                 # Dataset loading and preprocessing
├── utils/
│   ├── Condition_Encoder.py       # Conditional feature encoder
│   ├── Diffusion.py               # Diffusion process
│   └── TwoDenoiseModel.py         # Denoising network
├── TrainNetworkHelper_TwoTG.py    # Training utilities
├── Train_TwoTGCD_Conf.py          # Training entry point
└── Test_TwoTGCD_Conf.py           # Evaluation entry point
```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- tqdm
- einops
- OpenAI CLIP

Install the main dependencies with:

```bash
pip install torch torchvision numpy matplotlib tqdm einops
pip install git+https://github.com/openai/CLIP.git
```
