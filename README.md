# CosmoMamba

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19572872.svg)](https://doi.org/10.5281/zenodo.19572872)

**State-Space Models for Cosmological Parameter Inference**

First application of selective state-space models (Mamba) to cosmological field analysis, benchmarked against CNNs and Vision Transformers on the [CAMELS Multifield Dataset](https://camels-multifield-dataset.readthedocs.io/).

## Motivation

Inferring cosmological parameters (Omega_m, sigma_8) from 2D field maps of the universe is a key problem in observational cosmology. Current approaches use CNNs or Vision Transformers, but state-space models offer O(n) complexity vs O(n^2) for attention while capturing long-range spatial dependencies through multi-directional scanning. This work explores whether SSMs can match or exceed transformer/CNN performance on this task.

## Architecture

CosmoMamba processes 256x256 cosmological field maps through:
1. **Convolutional patch embedding** (7x7 → 3x3 with stride reduction)
2. **Stack of MambaBlocks**, each containing:
   - Multi-directional 2D scan (4 routes: row-forward, row-backward, column-forward, column-backward)
   - Selective SSM per direction
   - Learned merge projection
   - Feed-forward network
3. **Global average pooling → regression head** predicting mean + log-variance

## Quick Start

```bash
# 1. Create environment
conda create -n cosmo_mamba python=3.10
conda activate cosmo_mamba
pip install -r requirements.txt

# 2. Download data (dark matter density, ~3.75 GB per suite)
python download_data.py --field Mcdm --suite IllustrisTNG

# 3. Train CosmoMamba
python train.py --data_dir ./cmd_data

# 4. Train baselines for comparison
python train.py --config configs/cnn_baseline.yaml --data_dir ./cmd_data
python train.py --config configs/vit_baseline.yaml --data_dir ./cmd_data

# 5. Evaluate with plots
python evaluate.py --checkpoint runs/cosmo_mamba_Mcdm_TNG/best_model.pt

# 6. Cross-suite robustness test
python download_data.py --field Mcdm --suite SIMBA
python evaluate.py --checkpoint runs/cosmo_mamba_Mcdm_TNG/best_model.pt --cross_suite SIMBA
```

## Full Benchmark

```bash
# Run all models + cross-suite tests
bash run_experiments.sh ./cmd_data
```

## Project Structure

```
cosmo-mamba/
├── configs/
│   ├── default.yaml          # CosmoMamba config
│   ├── cnn_baseline.yaml     # CNN baseline config
│   └── vit_baseline.yaml     # ViT baseline config
├── data/
│   └── cmd_dataset.py        # CMD data loading + preprocessing
├── models/
│   ├── cnn_baseline.py       # CNN baseline (1.1M params)
│   ├── vit.py                # Vision Transformer (10.9M params)
│   └── cosmo_mamba.py        # CosmoMamba with parallel scan (4.0M params)
├── utils/
│   └── metrics.py            # R², RMSE, relative error
├── notebooks/
│   └── kaggle_cosmo_mamba.ipynb  # Self-contained Kaggle/Colab notebook
├── paper/
│   ├── main.tex              # Full paper (revtex4-2 format)
│   ├── references.bib        # BibTeX references
│   └── fig_*.png             # Result figures
├── train.py                  # Training loop with auto-resume
├── evaluate.py               # Evaluation + visualization
├── download_data.py          # CMD data download helper
├── run_experiments.sh        # Full benchmark runner
└── requirements.txt
```

## Data

Uses the [CAMELS Multifield Dataset (CMD)](https://camels-multifield-dataset.readthedocs.io/):
- 2D maps: 256x256 pixels, 13 field types (gas density, dark matter, temperature, etc.)
- Simulation suites: IllustrisTNG, SIMBA, Astrid
- Labels: Omega_m, sigma_8 (+ 4 astrophysical parameters)
- LH set: 1000 simulations x 15 maps = 15,000 maps per field per suite

## Results

### In-Domain Benchmark (IllustrisTNG Mcdm)

| Model | Params | R²(Ωm) | R²(σ8) | RMSE(Ωm) | RMSE(σ8) |
|-------|--------|---------|--------|-----------|----------|
| CNN | 1.1M | **0.993** | **0.978** | **0.010** | **0.018** |
| ViT | 10.9M | 0.938 | 0.818 | 0.030 | 0.052 |
| CosmoMamba | 4.0M | 0.936 | 0.886 | 0.030 | 0.041 |

### Cross-Suite Transfer (IllustrisTNG → SIMBA)

| Model | In-domain R² | Cross-suite R² | Δ |
|-------|-------------|----------------|---|
| CNN | 0.985 | 0.972 | -0.013 |
| ViT | 0.878 | 0.844 | -0.034 |
| CosmoMamba | 0.911 | 0.887 | **-0.024** |

CosmoMamba outperforms ViT on σ8 (R² = 0.886 vs 0.818) with 2.7x fewer parameters and O(n) complexity, and shows 30% less degradation than ViT in cross-suite transfer.

## Kaggle / Colab

A self-contained notebook with data download, training, and evaluation is provided in `notebooks/kaggle_cosmo_mamba.ipynb`. It includes auto-resume from checkpoints to handle session timeouts.

## Paper

The full paper is in `paper/`. To compile:

```bash
cd paper
pdflatex main.tex
# or upload the paper/ folder to Overleaf
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{dongre2025cosmomamba,
  title={CosmoMamba: State-Space Models for Cosmological Parameter Inference from 2D Field Maps},
  author={Dongre, Pratik},
  year={2025},
  doi={10.5281/zenodo.19572872},
  url={https://doi.org/10.5281/zenodo.19572872}
}
```

## References

- Villaescusa-Navarro et al., "The CAMELS Multifield Dataset" (2022) — [arXiv:2109.10915](https://arxiv.org/abs/2109.10915)
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Liu et al., "VMamba: Visual State Space Model" (2024) — [arXiv:2401.10166](https://arxiv.org/abs/2401.10166)

## License

MIT
