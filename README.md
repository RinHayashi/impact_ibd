# IMPACT-IBD

<p align="center">
  <img src="src/impact_ibd/streamlit_app/assets/logo.svg" alt="IMPACT-IBD logo" width="140">
</p>

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/RinHayashi/impact_ibd)

A Python package providing convenient access to **IMPACT-IBD** project data and tools for **AMG (Auxiliary Metabolic Gene) module scoring** and **sample classification** using pretrained models.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Install from GitHub Releases](#install-from-github-releases)
  - [Use as a Python package](#use-as-a-python-package)
  - [Run the Streamlit app](#run-the-streamlit-app)
- [Quick Start](#quick-start)
  - [Load bundled IMPACT-IBD data](#1-load-bundled-impact-ibd-data)
  - [Score and classify your own samples](#2-score--classify-your-own-samples)
- [Bundled Data](#bundled-data)
- [Pretrained Models](#pretrained-models)
- [API Reference](#api-reference)
- [Background](#background)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

<!-- TODO: 简要介绍 IMPACT-IBD 项目的科学背景与研究目标 -->

**IMPACT-IBD** is a phage auxiliary metabolic gene (AMG)-driven multi-omics metagenomic atlas of inflammatory bowel disease (IBD). This package bundles:

1. **Reference datasets** — curated abundance matrices and annotation tables from the IMPACT-IBD cohort.
2. **AMG scoring & classification** — pretrained Weighted-Distance KNN models that score three AMG modules (histidine, PTS/carbohydrate, flagellum) and classify external samples.

We recommend the Streamlit app as the visual interface for these package features. It provides a data browser for the reference datasets (1) and a way to try the pretrained scoring and classification models (2) without writing code.

> For full scientific context, cohort design, and biological interpretation, see [Background](#background) and [Citation](#citation).

---

## Features

- **Easy to install as a package and an app** — one install is enough to use IMPACT-IBD from Python or from the Streamlit app. The app interface is compact and easy to use for browsing bundled datasets and trying pretrained models. The package API is short, readable, and easy to extend, including one-line loaders for the built-in Parquet tables.
- **Pretrained classifiers** — ready-to-use models for:
  - `ibd_control` — IBD vs. healthy control
  - `cd_uc` — Crohn's disease (CD) vs. ulcerative colitis (UC)
- **Flexible evaluation** — predict on new data with or without ground-truth metadata; optional ROC/confusion-matrix PDF reports.

---

## Installation

Requires **Python 3.9**.

IMPACT-IBD is distributed as a wheel through GitHub Releases. A single installation provides both the Python package and the Streamlit app.

### Install from GitHub Releases

Create and activate a Python 3.9 Conda environment:

```bash
conda create -n impact-ibd-env -c conda-forge python==3.9 pip
conda activate impact-ibd-env
```

Install the wheel from GitHub Releases:

```bash
python -m pip install "https://github.com/RinHayashi/impact_ibd/releases/download/v0.1.0/impact_ibd-0.1.0-py3-none-any.whl"
```

### Use as a Python package

With the Conda environment activated, import the package in Python or a Jupyter notebook:

```python
import impact_ibd as imibd
```

### Run the Streamlit app

With the same environment activated, start the installed app:

```bash
impact-ibd-app
```

---

## Quick Start

### 1. Load bundled IMPACT-IBD data

```python
import impact_ibd as imibd

# AMG annotation table
amg_meta = imibd.load_amg_profile()

# AMG abundance matrix
amg_abd = imibd.load_amg_abd()
```

**After loading, set the first column as the row index** before using the table:

```python
amg_abd = imibd.load_amg_abd()
amg_abd = amg_abd.set_index(amg_abd.columns[0])

host = imibd.load_host_profile()
host = host.set_index(host.columns[0])
```

Available loaders:

| Function | Description | Rows | Columns |
|---|---|---|---|
| `load_amg_profile()` | AMG gene annotation | AMG ORF names (e.g. `PRJEB1220__ERR209533__k89_17697__full-cat_2_25`) | annotation columns |
| `load_amg_abd()` | AMG abundance matrix | AMG functions (e.g. `K00012`) | sample names |
| `load_votu_profile()` | vOTU profile | vOTU representative sequence names (e.g. `PRJEB1220__ERR209533__k89_17697\|\|full`) | annotation columns |
| `load_prok_profile()` | Prokaryote profile | taxid (e.g. `729`) | annotation columns |
| `load_host_profile()` | Sample metadata | sample names | metadata columns |

All functions return a `pandas.DataFrame`.

---

### 2. Score & classify your own samples

#### Prepare input data

Your abundance table and metadata must follow this layout:

| File | Format |
|---|---|
| Abundance (`X_test`) | TSS normalized abundance file.; **rows = sample IDs**, **columns = feature (gene) names** |
| Metadata (`meta_test`, optional) | **rows = sample IDs**; must include a target column for evaluation |

```python
import pandas as pd
from impact_ibd import amgscr

# Load user test set (sample names as row index, features as columns)
X_test = pd.read_csv("test_abundance_norm.csv", index_col=0)
meta_test = pd.read_csv("test_meta.csv", index_col=0)
```

#### Load a pretrained model

```python
model = amgscr.load_pretrained_model(task="ibd_control")
# Other built-in task: task="cd_uc"
# Or load a custom model: amgscr.load_pretrained_model(model_path="path/to/model.joblib")
```

#### Inspect nearest reference samples

Return the labeled reference neighbors used by the model for one or more test
samples:

```python
neighbors = model.get_nearest_neighbors(
    X_query=X_test.loc[["sample_01"]],
    topk=5,
)

# Columns: Query_Sample, Rank, Reference_Sample, Reference_Label,
#          Distance, Similarity
print(neighbors)
```

This method only reads the reference data already stored in the pretrained
model. It does not retrain or modify the model. Its distance and similarity
definitions are identical to `model.search()`.

#### Export AMG module scores

Uses the model's fitted scalers to compute Z-score-based module scores:

```python
module_scores = model.export_module_scores(
    train=False,
    X_test=X_test,
    output_path="AMG_module_score_testset.csv",
)
# Columns: SampleID, Dataset, His_Score, PTS_Score, Fla_Score
```

#### Predict (no ground truth)

When metadata is not available, only classification predictions are exported:

```python
model.evaluate(
    X_test=X_test,
    save_csv="predictions.csv",
)
```

#### Evaluate (with ground truth)

Provide metadata to compute metrics and optionally save a PDF report (ROC curve + confusion matrix):

```python
metrics = model.evaluate(
    X_test=X_test,
    meta_test=meta_test,
    target_col="Group2",       # column name in meta_test
    pos_label="IBD",           # positive class label
    threshold=0.77,            # decision threshold (optional; uses model default if omitted)
    save_csv="evaluation_results.csv",
    save_pdf="evaluation_report.pdf",
)

print(metrics)
# e.g. {'used_threshold': 0.77, 'pos_label': 'IBD',
#       'accuracy': ..., 'f1_score': ..., 'auc_roc': ..., ...}
```

> **Note:** Missing features in `X_test` are automatically filled with `0`; extra features are discarded. A feature-alignment report is printed during `evaluate()`.

---

## Bundled Data

<!-- TODO: 补充各数据表的样本量、特征维度、数据来源等详细信息 -->

Data files are shipped under `impact_ibd/data/` as Parquet:

| Dataset | Content |
|---|---|
| `AMG_profile.parquet` | AMG gene-level annotation |
| `AMG_abundance.parquet` | AMG gene abundance across IMPACT-IBD samples |
| `vOTU_profile.parquet` | vOTU-level profile |
| `host_profile.parquet` | Host / clinical sample metadata |
| `Prok_profile.parquet` | Prokaryote-level profile |

Loader tables keep the identifier in the **first column**. After `set_index(df.columns[0])`, the AMG abundance matrix is **features × samples** (see [Quick Start §1](#1-load-bundled-impact-ibd-data)). Scoring and classification expect **samples × features**, so transpose the bundled abundance if you use it as `X_test`:

```python
amg_abd = imibd.load_amg_abd()
amg_abd = amg_abd.set_index(amg_abd.columns[0])  # features × samples
X = amg_abd.T                                    # samples × features
```

---

## Pretrained Models

Models are stored in `impact_ibd/model/` as `.joblib` files:

| Task key | Model file | Description |
|---|---|---|
| `ibd_control` | `ibd_control_optimal.joblib` | IBD vs. Control |
| `cd_uc` | `cd_uc_optimal.joblib` | CD vs. UC subtyping |

Each model is a `WeightedDistanceKNN` instance fitted on the IMPACT-IBD reference cohort, with three weighted AMG modules:

| Module | Score column | Description |
|---|---|---|
| Amino acid (His) | `His_Score` | Histidine metabolism module |
| Carbohydrate (PTS) | `PTS_Score` | Phosphotransferase system module |
| Flagellum (Fla) | `Fla_Score` | Flagellar assembly module |

---

## API Reference

### Data loading (`impact_ibd`)

```python
import impact_ibd as imibd

df = imibd.load_amg_profile()   # returns pd.DataFrame
df = df.set_index(df.columns[0])
```

See [Quick Start §1](#1-load-bundled-impact-ibd-data) for the full list of loaders and the row/column layout after setting the index.

### Spherical 3D atlas (`impact_ibd.atlas`)

| Function | Description |
|---|---|
| `visualize_amg_atlas(amg_abd, host_profile, color_col, seed, output_dir, ...)` | End-to-end atlas: coordinates + spherical map + HTML figure |
| `compute_amg_atlas_coordinates(...)` | Global / module PCA, OOF disease scores, 3D PCA & UMAP ball coordinates |
| `add_spherical_coordinates(atlas, ...)` | Convert disease-aware UMAP axes to spherical XYZ |
| `plot_spherical_amg_atlas(atlas, color_col, ...)` | Depth-aware Plotly 3D figure (returns `plotly.graph_objects.Figure`) |

`color_col` is the metadata column used **only for colouring**. Disease-aware embeddings always use Control / UC / CD in `group_col` (auto-detected as `Group` or `group`).

### Scoring & classification (`impact_ibd.amgscr`)

| Function / Method | Description |
|---|---|
| `amgscr.load_pretrained_model(task, model_path)` | Load a built-in or custom pretrained model |
| `model.get_nearest_neighbors(X_query, topk, target_col, tau)` | Return reference sample names, labels, distances, and similarities |
| `model.export_module_scores(train, X_test, output_path)` | Compute and export AMG module scores |
| `model.evaluate(X_test, meta_test, ...)` | Predict and optionally evaluate with metrics |

For advanced usage (custom training, threshold optimization), see docstrings in `amgscr.py`.

---

## Background

<!-- TODO: 填写 IMPACT-IBD 项目的科学意义、研究设计、样本来源、AMG 模块选择的生物学依据等 -->

---

## Citation

<!-- TODO: 填写正式发表后的引用格式，例如：

If you use this package or the IMPACT-IBD data in your research, please cite:

> Author et al. (Year). Title. *Journal*. DOI: ...

-->

---

## License

<!-- TODO: 选择并填写许可证（如 MIT、Apache-2.0、CC-BY 等） -->

---

## Contact

**RinHayashi** — [ravensoch0122@gmail.com](mailto:ravensoch0122@gmail.com)

Issues and pull requests are welcome on [GitHub](https://github.com/RinHayashi/impact_ibd/issues).
