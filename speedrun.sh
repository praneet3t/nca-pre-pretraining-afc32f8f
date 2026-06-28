#!/bin/bash
# =============================================================================
# Minimal end-to-end reproduction of arXiv 2603.10055
#   "Training Language Models via Neural Cellular Automata"
#
# Core claim (minimal slice): a transformer pre-pre-trained on NCA synthetic
# data, then trained on natural language, reaches lower validation perplexity
# than an identical model trained from scratch.
#
# Pipeline:
#   1. NCA pre-pre-training  (src/nca_ppt.py, --token patch tokenization)
#   2. OpenWebText transfer  (src/openwebtext_pt.py, load NCA ckpt, reinit embed)
#   3. OpenWebText scratch   (src/openwebtext_pt.py, random init)
#   4. Summarize -> EVAL.md  (perplexity + convergence comparison)
#
# Everything is sized small so it runs on a single GPU in a handful of minutes.
# =============================================================================
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
# keep JAX (NCA data generation) on CPU so it doesn't contend with torch for GPU memory
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=0

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

WORK=${WORK:-/tmp/nca_minrepro}
ART="$ROOT/.openresearch/artifacts"
mkdir -p "$WORK" "$ART"

NCA_DIR="$WORK/nca_ckpt"
OWT_DATA="$WORK/owt_data"
OWT_NCA_DIR="$WORK/owt_from_nca"
OWT_SCR_DIR="$WORK/owt_scratch"
mkdir -p "$NCA_DIR" "$OWT_DATA" "$OWT_NCA_DIR" "$OWT_SCR_DIR"

# ---- shared tiny architecture -------------------------------------------------
N_LAYER=4
N_HEAD=4
N_EMBD=256
SEQ=256
PT_VOCAB=64000          # NCA patch vocab (10^4) fits inside this; matches OWT core

echo "==============================================================="
echo " STEP 0: install dependencies"
echo "==============================================================="
# Install only what the minimal repro needs (the full pinned requirements.txt
# includes packages like mkl-service that aren't installable here). torch is
# preinstalled on the box; we keep it as-is.
python -m pip install -q \
    "jax[cpu]==0.6.2" "jaxlib==0.6.2" flax==0.11.2 optax==0.2.5 einops \
    tiktoken datasets==3.6.0 transformers==4.53.0 numpy matplotlib tqdm \
    huggingface-hub safetensors wandb peft joblib 2>&1 | tail -5
# jax CPU is enough for NCA data generation; torch uses the GPU.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import jax, tiktoken, datasets, transformers, flax; print('deps ok')"

echo "==============================================================="
echo " STEP 1: prepare a small OpenWebText slice (train.bin / val.bin)"
echo "==============================================================="
if [ ! -f "$OWT_DATA/train.bin" ]; then
  python minrepro/prepare_owt.py \
    --out_dir "$OWT_DATA" \
    --train_tokens 6000000 \
    --val_tokens 300000
fi

echo "==============================================================="
echo " STEP 2: NCA pre-pre-training (tiny transformer on NCA dynamics)"
echo "==============================================================="
python src/nca_ppt.py \
    --seed 0 \
    --device cuda:0 \
    --grid 12 --patch 2 --num_colors 10 \
    --seq_len $SEQ \
    --token \
    --vocab_size $PT_VOCAB \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --num_epochs 3 \
    --warmup 1 \
    --model_type llama \
    --model_name llama-mini \
    --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD \
    --temperature 1e-3 \
    --train_num_rules 256 --val_num_rules 64 \
    --train_num_sim 4 --val_num_sim 2 \
    --dT 1 \
    --init_rollout_steps 10 \
    --filter_rules --filter_rules_threshold 0.5 \
    --filter_rules_upper_bound 1.0 --filter_rules_mode gzip \
    --generate_train --generate_rules 1 \
    --val_freq 50 \
    --mixed_precision bf16 \
    --save_dir "$NCA_DIR" \
    --interval_save --intervals 1 2 3

# pick the latest NCA checkpoint file
NCA_CKPT=$(ls -t "$NCA_DIR"/model_*.pth 2>/dev/null | head -1 || true)
if [ -z "${NCA_CKPT:-}" ]; then
  NCA_CKPT=$(ls -t "$NCA_DIR"/best_model_*.pth | head -1)
fi
NCA_CKPT_FILE=$(basename "$NCA_CKPT")
echo "Using NCA checkpoint: $NCA_CKPT_FILE"

# ---- shared OWT training hyperparameters --------------------------------------
OWT_COMMON=(
  --device 0
  --seed 0
  --data_dir "$OWT_DATA"
  --model_type llama
  --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD
  --pt_vocab_size $PT_VOCAB
  --pt_seq_len $SEQ --new_seq_len $SEQ
  --batch_size 16
  --gradient_accumulation_steps 2
  --lr 5e-4
  --epochs 1
  --warmup 0.05
  --val_freq 25
  --mixed_precision bf16
  --weight_decay 0.0001
  --grad_clip_enable 1 --grad_clip 1.0
)

echo "==============================================================="
echo " STEP 3a: OpenWebText training FROM NCA checkpoint (transfer)"
echo "==============================================================="
python src/openwebtext_pt.py \
    "${OWT_COMMON[@]}" \
    --save_dir "$OWT_NCA_DIR" \
    --pretrain 1 \
    --model_path "$NCA_DIR" \
    --model_file "$NCA_CKPT_FILE" \
    --reinit_modules embed none

echo "==============================================================="
echo " STEP 3b: OpenWebText training FROM SCRATCH (control)"
echo "==============================================================="
python src/openwebtext_pt.py \
    "${OWT_COMMON[@]}" \
    --save_dir "$OWT_SCR_DIR" \
    --pretrain 0

echo "==============================================================="
echo " STEP 4: summarize -> EVAL.md"
echo "==============================================================="
python minrepro/summarize.py \
    --nca_metrics "$OWT_NCA_DIR/metrics.json" \
    --scratch_metrics "$OWT_SCR_DIR/metrics.json" \
    --eval_out "$ROOT/EVAL.md" \
    --artifacts_dir "$ART"

echo "DONE. EVAL.md written."
