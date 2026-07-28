# VGTA chlorophyll-a reconstruction

Cheng Xu, Ningxia University

This repository implements the reproducibility workflow for remote-sensing-assisted reconstruction of four-hourly chlorophyll-a dynamics. It contains Sentinel-3 OLCI product validation and feature construction, a twenty-channel Vision Mamba retrieval model, seeded GKOA hyperparameter search, mask-aware TCN-Attention reconstruction, and classification and reconstruction metrics.

The final time series is a model reconstruction constrained by in-situ observations and daily satellite priors. It is not a direct four-hourly satellite observation product.

## Method

The ordered workflow is:

1. identify acquisition-matched Sentinel-3 products;
2. validate the Sentinel-3 OLCI-1 EFR and OLCI-2 LFR products named in the manuscript;
3. construct the ordered twenty-variable OLCI feature cube from the released study arrays;
4. train Vision Mamba on the eighteen source lakes;
5. generate daily Chl-a concentrations and state predictions;
6. optimize GTA learning rate, kernel size, and filter count on four evolutionary-validation lakes;
7. reconstruct masked four-hourly sequences with explicit observation and satellite-prior masks;
8. evaluate the five untouched target lakes.

Vision Mamba uses spatial patch embedding, positional embedding, bidirectional selective state-space scans, local convolution, and gating. Its default 20-channel configuration has 8,823,221 trainable parameters. GTA uses causal dilated temporal convolution and two-head temporal attention over continuous multi-day four-hour sequences. Original observations are retained; the model fills only missing positions.

## Chlorophyll-a states

Class labels, the number of classes, and the Bloom-F1 positive label are supplied with the released study data. The code does not redefine them.

## Environment

```bash
conda env create -f environment.yml
conda activate vgta-repro
python -m pip install --no-build-isolation -e .
```

The locked reproduction environment uses Python 3.8.2. No CUDA version is asserted because the manuscript and supporting material do not report one.

## Data

Raw observations are not committed to this code repository. The expected external layout, aligned array schema, official sources, and product requirements are described in (data/README.md).

The executable training pipeline consumes an authorized `prepared.npz` file containing lake-disjoint source, evolutionary-validation, and target arrays. Product functions in `vgta.inventory` and `vgta.olci` validate the EFR/LFR evidence files. Feature equations are implemented in `vgta.feature_engineering`.


## Outputs

The workflow writes ViM classification metrics and checkpoint, the complete GKOA evaluation history, GTA reconstruction metrics and checkpoint, target predictions and supervision masks, and a SHA-256 input manifest. Accuracy, Macro-F1, Bloom-F1, confusion matrices, MAE, RMSE, MAPE, sMAPE (%), MdAE, NSE, Pearson R-squared, and log-MAE are calculated from saved targets and predictions.

No article result is stored as an executable constant. If a fresh result differs from the article, the fresh result and its manifest take precedence for reproducibility reporting.

## Attribution and license

The Vision Mamba architecture follows the method introduced by Zhu et al. and the public Vim project: https://github.com/hustvl/Vim. The implementation in this repository is self-contained and does not vendor the upstream detection or segmentation projects.

