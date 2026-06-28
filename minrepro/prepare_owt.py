"""
Minimal OpenWebText binarizer for the minimal repro.

Downloads a small slice of OpenWebText, tokenizes with the GPT-2 (tiktoken)
encoding, and writes train.bin / val.bin (uint16) in the format expected by
utils.dataset_utils.OpenWebTextDataset.

Kept deliberately tiny so the whole pipeline runs end-to-end on one GPU in
a few minutes. Token budget is controlled by --train_tokens / --val_tokens.
"""
import os
import sys
import argparse

import numpy as np
import tiktoken
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_tokens", type=int, default=6_000_000)
    ap.add_argument("--val_tokens", type=int, default=300_000)
    ap.add_argument("--dataset", type=str, default="stas/openwebtext-10k",
                    help="HF dataset id; default is a small 10k-doc OWT slice")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print(f"[prepare_owt] loading {args.dataset} ...", flush=True)
    ds = load_dataset(args.dataset, split="train")
    print(f"[prepare_owt] {len(ds)} docs", flush=True)

    target = {"train": args.train_tokens, "val": args.val_tokens}
    # val first (from the tail), train from the head, so they don't overlap.
    n = len(ds)
    val_docs = ds.select(range(max(0, n - 500), n))
    train_docs = ds.select(range(0, n - 500))

    def write_split(name, docs, n_tokens):
        path = os.path.join(args.out_dir, f"{name}.bin")
        buf = []
        total = 0
        for ex in docs:
            ids = enc.encode_ordinary(ex["text"])
            ids.append(eot)
            buf.extend(ids)
            total += len(ids)
            if total >= n_tokens:
                break
        arr = np.array(buf[:n_tokens], dtype=np.uint16)
        arr.tofile(path)
        print(f"[prepare_owt] wrote {len(arr):,} tokens -> {path}", flush=True)

    write_split("val", val_docs, target["val"])
    write_split("train", train_docs, target["train"])
    print("[prepare_owt] done", flush=True)


if __name__ == "__main__":
    main()
