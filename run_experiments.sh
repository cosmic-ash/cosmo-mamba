#!/bin/bash
# Full benchmark: CNN vs ViT vs CosmoMamba on CMD dark matter density maps
# Run this on a machine with a GPU (Colab, Lambda, etc.)

set -e

echo "============================================"
echo "CosmoMamba Benchmark Suite"
echo "============================================"

DATA_DIR="${1:-./cmd_data}"

# Step 1: Download data (dark matter density from IllustrisTNG and SIMBA)
echo -e "\n[1/5] Downloading CMD data..."
python download_data.py --field Mcdm --suite IllustrisTNG SIMBA --output_dir "$DATA_DIR"

# Step 2: Train CNN baseline
echo -e "\n[2/5] Training CNN baseline..."
python train.py --config configs/cnn_baseline.yaml --data_dir "$DATA_DIR"

# Step 3: Train ViT baseline
echo -e "\n[3/5] Training ViT baseline..."
python train.py --config configs/vit_baseline.yaml --data_dir "$DATA_DIR"

# Step 4: Train CosmoMamba
echo -e "\n[4/5] Training CosmoMamba..."
python train.py --config configs/default.yaml --data_dir "$DATA_DIR"

# Step 5: Cross-suite robustness (test each model on SIMBA)
echo -e "\n[5/5] Cross-suite robustness evaluation..."
for model_dir in runs/cnn_baseline_* runs/vit_baseline_* runs/cosmo_mamba_*; do
    if [ -f "$model_dir/best_model.pt" ]; then
        echo "  Testing $model_dir on SIMBA..."
        python evaluate.py --checkpoint "$model_dir/best_model.pt" --cross_suite SIMBA --data_dir "$DATA_DIR"
    fi
done

echo -e "\n============================================"
echo "All experiments complete! Results in runs/"
echo "============================================"
