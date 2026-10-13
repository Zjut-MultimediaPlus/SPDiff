# Data acquisition and preparation

This document explains how to download and prepare PDTC and SETCD-Sat for end-to-end SPDiff training.

**Baidu Netdisk share link:** 通过网盘分享的文件：SPDiff-Data-Resource
链接: https://pan.baidu.com/s/1is4kguE8WYXplZhi35dMfw?pwd=hutu 提取码: hutu 
--来自百度网盘超级会员v1的分享

## Netdisk folder layout

```text
SPDiff-Data-Resource/
├── DATA_README.md
├── PDTC/
│   ├── Satellite_data/PDTC_Satellite_Data.tar.gz
│   └── Metadata/
│       ├── satellite_ir_norm_values_split.pkl
│       └── AllRadii_Labels.pkl
├── SETCD-Sat/
│   ├── Satellite_data/SETCD_Gridsat_Data.tar.gz
│   └── Metadata/
│       ├── ibtracs_age_hours.pkl
│       ├── ibtracs_row_dict.pkl
│       ├── ibtracs_columns.pkl
│       └── nan_over_half_first_channel.pkl
└── concept_embs_clip.pt
```

`Satellite_data/` contains satellite image archives. `Metadata/` contains labels, normalization statistics, storm ages, column definitions and invalid-sample records. Preserve these filenames and their capitalization.

## Concept Embeddings

We provide concept embeddings for SPDiff training. The predefined concepts described in the paper are encoded using OpenAI CLIP ViT-B/32. The 40 concepts in seven categories are represented by 512-dimensional, L2-normalized embeddings. Both PDTC and SETCD-Sat use the embeddings provided in `SPDiff-Data-Resource/concept_embs_clip.pt`. The file contains a dictionary with the key `concept_embs` and is loaded directly during training.

## Prepare downloaded files

Download the whole resource folder and upload it to your server's data storage location. Keep metadata and concept embeddings in their downloaded locations. Only the satellite archives need extraction; no file copying or PKL conversion is required. The example below uses `/path/to/SPDiff-Data-Resource`; replace it with your actual server path.

```bash
export SPDIFF_DATA_ROOT=/path/to/SPDiff-Data-Resource

# Extract PDTC.
tar -xzf "$SPDIFF_DATA_ROOT/PDTC/Satellite_data/PDTC_Satellite_Data.tar.gz" \
  -C "$SPDIFF_DATA_ROOT/PDTC/Satellite_data"

# Extract SETCD-Sat.
tar -xzf "$SPDIFF_DATA_ROOT/SETCD-Sat/Satellite_data/SETCD_Gridsat_Data.tar.gz" \
  -C "$SPDIFF_DATA_ROOT/SETCD-Sat/Satellite_data"
```

The commands preserve the `k89_4ch1/` and `Gridsat_IR/` folders contained in the PDTC and SETCD-Sat archives, respectively. After extraction, the data directories should be:

```text
SPDiff-Data-Resource/
├── PDTC/Satellite_data/k89_4ch1/
│   ├── train/
│   ├── valid/
│   └── test/
└── SETCD-Sat/Satellite_data/Gridsat_IR/
    ├── train/
    ├── valid/
    └── test/
```

Keep the supplied splits and NPY filenames. The loaders accept cyclone names and timestamps used by the prepared data; PDTC uses YYYYMMDDHH and SETCD-Sat uses YYYYMMDDHHMM. Rotation suffixes are retained.

## Configure the data root and dataset

In `env/Config.py`, set `dataset_name` to `PDTC` or `SETCD-Sat` and `DATA_ROOT` to the actual server resource folder. Alternatively, set the environment variables:

```bash
export SPDIFF_DATA_ROOT=/path/to/SPDiff-Data-Resource
export SPDIFF_DATASET=PDTC       # Or SETCD-Sat
```

All metadata, split, concept-embedding and model input paths/settings are selected automatically. There is no need to edit individual paths or switch code files.

See [README.md](README.md#train-and-evaluate) for each dataset's training and evaluation commands.
