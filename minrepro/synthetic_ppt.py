"""
Generic synthetic pre-pre-training for non-NCA sources (sudoku, random).

Builds the SAME tiny Llama as nca_ppt.py (so checkpoints transfer identically),
trains next-token prediction on the chosen source, and saves a checkpoint in
the format openwebtext_pt.py's load_model expects (model_<epoch>.pth).
"""
import os
import sys
import argparse
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

from utils.models import create_llama_model, create_attention_mask
from utils.util import set_seed, get_lr_scheduler, save_checkpoint, delete_old_checkpoint, setup_logger
from minrepro.synthetic_sources import get_dataset, SYNTH_VOCAB

log = setup_logger("synthetic_ppt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["sudoku", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=64000)  # match OWT model dims
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--num_epochs", type=int, default=12)
    ap.add_argument("--warmup", type=float, default=1)
    ap.add_argument("--num_sequences", type=int, default=256)  # per epoch
    ap.add_argument("--val_sequences", type=int, default=64)
    ap.add_argument("--val_freq", type=int, default=50)
    ap.add_argument("--mixed_precision", type=str, default="none", choices=["bf16", "fp16", "none"])
    ap.add_argument("--save_dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    autocast = args.mixed_precision != "none"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision, None)

    log.info(f"[{args.source}] device={device} seq_len={args.seq_len}")
    log.info(f"[{args.source}] building model ({args.n_layer}L/{args.n_embd}d)")

    model = create_llama_model(
        vocab_size=args.vocab_size, seq_length=args.seq_len,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        output_vocab=args.vocab_size,
    ).to(device)
    log.info(f"[{args.source}] params={sum(p.numel() for p in model.parameters()):,}")

    train_ds = get_dataset(args.source, args.num_sequences, args.seq_len, seed=args.seed)
    val_ds = get_dataset(args.source, args.val_sequences, args.seq_len, seed=args.seed + 999)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    log.info(f"[{args.source}] train batches={len(train_dl)} val batches={len(val_dl)}")

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=len(train_dl) * args.warmup // args.grad_accumulation_steps,
        total_steps=len(train_dl) * args.num_epochs // args.grad_accumulation_steps,
    )

    base_mask = create_attention_mask(args.seq_len, additive=True).to(device)
    metrics = defaultdict(list)
    best_val = float("inf")
    total_iters = 0

    for epoch in range(args.num_epochs):
        model.train()
        acc_loss = 0.0
        acc_step = 0
        for seq, targets in train_dl:
            seq = seq.to(device)
            targets = targets.to(device)
            B = seq.shape[0]
            attn = base_mask.repeat(B, 1, 1, 1)
            with torch.autocast(device_type=device.type, enabled=autocast, dtype=autocast_dtype):
                logits = model(seq, attention_mask=attn)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            acc_loss += loss.detach().item()
            acc_step += 1
            if autocast:
                scaler = getattr(main, "_scaler", None)
                if scaler is None:
                    scaler = torch.amp.GradScaler(device.type)
                    main._scaler = scaler
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if acc_step % args.grad_accumulation_steps == 0:
                if autocast:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                total_iters += 1

        avg = acc_loss / max(1, acc_step)
        # validation
        vloss, vcount = 0.0, 0
        model.eval()
        with torch.no_grad():
            for seq, targets in val_dl:
                seq = seq.to(device); targets = targets.to(device)
                B = seq.shape[0]
                attn = base_mask.repeat(B, 1, 1, 1)
                with torch.autocast(device_type=device.type, enabled=autocast, dtype=autocast_dtype):
                    logits = model(seq, attention_mask=attn)
                    vloss += criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
                    vcount += 1
        vloss = vloss / max(1, vcount)
        log.info(f"[{args.source}] epoch {epoch+1}/{args.num_epochs} train_loss={avg:.4f} val_loss={vloss:.4f}")
        metrics["train/loss"].append(avg)
        metrics["val/loss"].append(vloss)
        if vloss < best_val:
            best_val = vloss
            save_checkpoint(epoch+1, 0, model, optimizer, scheduler, best_val, best_val, metrics, args.save_dir, best=True)
        save_checkpoint(epoch+1, 0, model, optimizer, scheduler, best_val, best_val, metrics, args.save_dir)
        delete_old_checkpoint(args.save_dir)
    log.info(f"[{args.source}] done. best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
