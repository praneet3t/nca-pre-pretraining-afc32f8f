"""
Read the val-loss curves from the NCA-pretrained and scratch OpenWebText runs,
compute perplexity, convergence speed-up, and write EVAL.md + a comparison
JSON/CSV into .openresearch/artifacts/.

Core claim under test (paper 2603.10055): a transformer pre-pre-trained on NCA
synthetic data, then trained on natural language, reaches *lower* validation
perplexity and converges *faster* than an identical model trained from scratch.
"""
import os
import json
import math
import argparse


def load_curve(metrics_path):
    """Return list of (iteration, val_loss) from a metrics.json file."""
    with open(metrics_path) as f:
        m = json.load(f)
    curve = []
    for entry in m.get("val/loss_mean", []):
        # each entry is {"<iter>": loss}
        for k, v in entry.items():
            curve.append((int(k), float(v)))
    curve.sort()
    return curve


def best(curve):
    return min(v for _, v in curve) if curve else float("nan")


def tokens_to_reach(curve, target):
    """First iteration at which val_loss <= target (None if never)."""
    for it, v in curve:
        if v <= target:
            return it
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nca_metrics", required=True)
    ap.add_argument("--scratch_metrics", required=True)
    ap.add_argument("--eval_out", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    nca = load_curve(args.nca_metrics)
    scr = load_curve(args.scratch_metrics)

    nca_best = best(nca)
    scr_best = best(scr)
    nca_ppl = math.exp(nca_best) if nca_best == nca_best else float("nan")
    scr_ppl = math.exp(scr_best) if scr_best == scr_best else float("nan")

    ppl_improve = (scr_ppl - nca_ppl) / scr_ppl * 100 if scr_ppl else float("nan")

    # Convergence: tokens (iterations) the NCA model needs to reach the scratch
    # model's *final* val loss, vs. the scratch model itself.
    scr_final = scr[-1][1] if scr else float("nan")
    nca_reach = tokens_to_reach(nca, scr_final)
    scr_reach = tokens_to_reach(scr, scr_final)
    speedup = (scr_reach / nca_reach) if (nca_reach and scr_reach) else None

    summary = {
        "nca_best_val_loss": nca_best,
        "scratch_best_val_loss": scr_best,
        "nca_best_ppl": nca_ppl,
        "scratch_best_ppl": scr_ppl,
        "ppl_improvement_pct": ppl_improve,
        "scratch_final_val_loss": scr_final,
        "nca_iters_to_scratch_final": nca_reach,
        "scratch_iters_to_scratch_final": scr_reach,
        "convergence_speedup_x": speedup,
        "nca_curve": nca,
        "scratch_curve": scr,
    }
    with open(os.path.join(args.artifacts_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV of the two curves for plotting later.
    with open(os.path.join(args.artifacts_dir, "val_curves.csv"), "w") as f:
        f.write("iteration,nca_val_loss,scratch_val_loss\n")
        d_nca = dict(nca)
        d_scr = dict(scr)
        for it in sorted(set(d_nca) | set(d_scr)):
            f.write(f"{it},{d_nca.get(it,'')},{d_scr.get(it,'')}\n")

    claim_supported = (nca_ppl < scr_ppl) if (nca_ppl == nca_ppl and scr_ppl == scr_ppl) else False
    verdict = "SUPPORTED" if claim_supported else "NOT SUPPORTED"

    lines = []
    lines.append("# Minimal reproduction: NCA pre-pre-training (arXiv 2603.10055)\n")
    lines.append(f"**Core claim verdict: {verdict}**\n")
    lines.append(
        "Claim: a transformer pre-pre-trained on Neural Cellular Automata (NCA) "
        "synthetic data, then trained on natural language (OpenWebText), reaches "
        "lower validation perplexity and converges faster than an identical model "
        "trained from scratch.\n"
    )
    lines.append("## Setup (minimal)\n")
    lines.append(
        "- Tiny Llama transformer (see speedrun.sh for exact size).\n"
        "- Stage 1: NCA pre-pre-training on on-the-fly generated NCA trajectories "
        "(`--token` patch tokenization).\n"
        "- Stage 2a (NCA): transfer all weights except embeddings, train on a small "
        "OpenWebText slice.\n"
        "- Stage 2b (scratch): identical training, random init.\n"
    )
    lines.append("## Results\n")
    lines.append("| Metric | NCA pre-pre-trained | Scratch |")
    lines.append("|---|---|---|")
    lines.append(f"| Best val loss | {nca_best:.4f} | {scr_best:.4f} |")
    lines.append(f"| Best val perplexity | {nca_ppl:.2f} | {scr_ppl:.2f} |")
    lines.append("")
    lines.append(f"- **Perplexity improvement (NCA vs scratch): {ppl_improve:.2f}%**")
    if speedup is not None:
        lines.append(
            f"- **Convergence speed-up: {speedup:.2f}x** "
            f"(NCA reached scratch's final loss {scr_final:.4f} in {nca_reach} iters "
            f"vs {scr_reach} iters for scratch)"
        )
    else:
        lines.append(
            f"- Convergence: NCA reached scratch's final loss ({scr_final:.4f}) at "
            f"iter {nca_reach}; scratch at iter {scr_reach}."
        )
    lines.append("")
    lines.append(
        "Artifacts: `comparison.json` (full numbers + curves), "
        "`val_curves.csv` (per-iteration val loss for both runs)."
    )
    lines.append("")

    with open(args.eval_out, "w") as f:
        f.write("\n".join(lines))
    # also drop a copy into artifacts
    with open(os.path.join(args.artifacts_dir, "EVAL.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
