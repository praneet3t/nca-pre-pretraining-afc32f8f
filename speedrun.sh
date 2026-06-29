#!/bin/bash
# =============================================================================
# Multi-source pre-pre-training comparison (arXiv 2603.10055)
#
# Tests the paper's "structure over semantics" thesis: does pre-pre-training on
# STRUCTURED synthetic data (NCA dynamics, Sudoku grids) transfer to language,
# while UNSTRUCTURED (random digits) and Scratch do not?
#
# Stages:
#   0. install deps
#   1. prepare a small OWT-style slice (train.bin / val.bin)
#   2. pre-pre-train three sources on the SAME tiny Llama:
#        a) NCA      (structured cellular-automata dynamics)
#        b) Sudoku   (structured human-rule 9x9 solved grids, digit alphabet)
#        c) Random   (i.i.d. digits 1-9 -- same alphabet as Sudoku, no structure)
#   3. for each source: transfer (reinit embeddings) -> short OWT training
#   4. scratch control: same OWT training, random init
#   5. summarize -> EVAL.md (+ comparison.json, val_curves.csv)
# =============================================================================
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=0

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

WORK=${WORK:-/tmp/nca_minrepro}
ART="$ROOT/.openresearch/artifacts"
mkdir -p "$WORK" "$ART"

OWT_DATA="$WORK/owt_data"
mkdir -p "$OWT_DATA"

# ---- shared tiny architecture -------------------------------------------------
N_LAYER=4
N_HEAD=4
N_EMBD=256
SEQ=256
PT_VOCAB=64000

echo "==============================================================="
echo " STEP 0: install dependencies"
echo "==============================================================="
python -m pip install -q \
    "jax[cpu]==0.6.2" "jaxlib==0.6.2" flax==0.11.2 optax==0.2.5 einops \
    tiktoken datasets==3.6.0 transformers==4.53.0 numpy matplotlib tqdm \
    huggingface-hub safetensors wandb peft joblib 2>&1 | tail -5
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import jax, tiktoken, datasets, transformers, flax; print('deps ok')"

echo "==============================================================="
echo " STEP 1: prepare a small OWT slice (short phase, 1M tokens)"
echo "==============================================================="
if [ ! -f "$OWT_DATA/train.bin" ]; then
  python minrepro/prepare_owt.py \
    --out_dir "$OWT_DATA" \
    --train_tokens 1000000 \
    --val_tokens 300000
fi

echo "==============================================================="
echo " STEP 2a: NCA pre-pre-training (structured dynamics)"
echo "==============================================================="
NCA_DIR="$WORK/nca_ckpt"
mkdir -p "$NCA_DIR"
python src/nca_ppt.py \
    --seed 0 --device cuda:0 \
    --grid 12 --patch 2 --num_colors 10 \
    --seq_len $SEQ --token --vocab_size $PT_VOCAB \
    --batch_size 16 --grad_accumulation_steps 1 \
    --learning_rate 3e-4 --num_epochs 12 --warmup 1 \
    --model_type llama --model_name llama-mini \
    --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD \
    --temperature 1e-3 \
    --train_num_rules 256 --val_num_rules 32 \
    --train_num_sim 16 --val_num_sim 8 \
    --dT 1 --init_rollout_steps 10 \
    --filter_rules --filter_rules_threshold 0.5 \
    --filter_rules_upper_bound 1.0 --filter_rules_mode gzip \
    --generate_train --generate_rules 1 \
    --val_freq 50 \
    --mixed_precision bf16 \
    --save_dir "$NCA_DIR" \
    --interval_save --intervals 1 2 3 4 6 8 10 12
NCA_CKPT=$(basename "$(ls -t "$NCA_DIR"/model_*.pth 2>/dev/null | head -1 || ls -t "$NCA_DIR"/best_model_*.pth | head -1)")

echo "==============================================================="
echo " STEP 2b: Sudoku pre-pre-training (structured human-rule grids)"
echo "==============================================================="
SUD_DIR="$WORK/sudoku_ckpt"; mkdir -p "$SUD_DIR"
python minrepro/synthetic_ppt.py \
    --source sudoku --seed 0 --device cuda:0 \
    --seq_len $SEQ --vocab_size $PT_VOCAB \
    --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD \
    --batch_size 16 --grad_accumulation_steps 1 \
    --learning_rate 3e-4 --num_epochs 12 --warmup 1 \
    --num_sequences 256 --val_sequences 64 --val_freq 50 \
    --mixed_precision bf16 --save_dir "$SUD_DIR"
SUD_CKPT=$(basename "$(ls -t "$SUD_DIR"/model_*.pth 2>/dev/null | head -1 || ls -t "$SUD_DIR"/best_model_*.pth | head -1)")

echo "==============================================================="
echo " STEP 2c: Random-digit pre-pre-training (unstructured control)"
echo "==============================================================="
RND_DIR="$WORK/random_ckpt"; mkdir -p "$RND_DIR"
python minrepro/synthetic_ppt.py \
    --source random --seed 0 --device cuda:0 \
    --seq_len $SEQ --vocab_size $PT_VOCAB \
    --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD \
    --batch_size 16 --grad_accumulation_steps 1 \
    --learning_rate 3e-4 --num_epochs 12 --warmup 1 \
    --num_sequences 256 --val_sequences 64 --val_freq 50 \
    --mixed_precision bf16 --save_dir "$RND_DIR"
RND_CKPT=$(basename "$(ls -t "$RND_DIR"/model_*.pth 2>/dev/null | head -1 || ls -t "$RND_DIR"/best_model_*.pth | head -1)")

# ---- shared OWT transfer hyperparameters --------------------------------------
OWT_COMMON=(
  --device 0 --seed 0
  --data_dir "$OWT_DATA"
  --model_type llama
  --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD
  --pt_vocab_size $PT_VOCAB
  --pt_seq_len $SEQ --new_seq_len $SEQ
  --batch_size 16 --gradient_accumulation_steps 2
  --lr 5e-4 --epochs 1 --warmup 0.05 --val_freq 25
  --mixed_precision bf16 --weight_decay 0.0001
  --grad_clip_enable 1 --grad_clip 1.0
)

run_transfer () {  # $1=name $2=ckpt_dir $3=ckpt_file
  local name="$1" cdir="$2" cfile="$3"
  local out="$WORK/owt_from_${name}"; mkdir -p "$out"
  echo "==============================================================="
  echo " STEP 3: OWT transfer FROM $name ($cfile)"
  echo "==============================================================="
  python src/openwebtext_pt.py \
      "${OWT_COMMON[@]}" \
      --save_dir "$out" \
      --pretrain 1 \
      --model_path "$cdir" --model_file "$cfile" \
      --reinit_modules embed none
}

run_transfer nca    "$NCA_DIR" "$NCA_CKPT"
run_transfer sudoku "$SUD_DIR" "$SUD_CKPT"
run_transfer random "$RND_DIR" "$RND_CKPT"

echo "==============================================================="
echo " STEP 4: OWT SCRATCH control"
echo "==============================================================="
SCR_DIR="$WORK/owt_scratch"; mkdir -p "$SCR_DIR"
python src/openwebtext_pt.py \
    "${OWT_COMMON[@]}" \
    --save_dir "$SCR_DIR" \
    --pretrain 0

echo "==============================================================="
echo " STEP 5: summarize -> EVAL.md"
echo "==============================================================="
python minrepro/summarize.py \
    --source "nca:$WORK/owt_from_nca/metrics.json" \
    --source "sudoku:$WORK/owt_from_sudoku/metrics.json" \
    --source "random:$WORK/owt_from_random/metrics.json" \
    --source "scratch:$WORK/owt_scratch/metrics.json" \
    --eval_out "$ROOT/EVAL.md" \
    --artifacts_dir "$ART"

echo "DONE. EVAL.md written."
