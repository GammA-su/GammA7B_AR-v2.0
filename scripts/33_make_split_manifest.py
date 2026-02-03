#!/usr/bin/env python3
"""
Generate a pack manifest that samples from per-domain split files.

Usage:
  uv run python scripts/33_make_split_manifest.py \
    --out configs/stageA_manifest_dedup_split.yaml \
    --split-rel data/dedup/stageA_latest_split_by_domain \
    --tokenizer-var TOKENIZER_JSON
"""
from __future__ import annotations
import argparse
import os
import json

STAGEA_WEIGHTS_PCT = {
    "filtered_web": 30.0,
    "books_longform": 18.0,
    "reference": 8.0,
    "forums_qa": 10.0,
    "code_docs": 18.0,
    "math": 12.0,
    "tutorials_notebooks": 2.0,
    "paper_teasers": 2.0,
}

def _normalized_weights(d: dict) -> dict:
    total = float(sum(d.values()))
    if total <= 0:
        raise ValueError(f"Total weight must be > 0, got {total}")
    return {k: (float(v) / total) for k, v in d.items()}

def _load_yield_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _apply_yield_correction(weights: dict, yield_data: dict) -> dict:
    if not yield_data:
        return dict(weights)
    corrected = {}
    for dom, w in weights.items():
        yd = yield_data.get(dom)
        if yd is None:
            corrected[dom] = w
            continue
        avg = yd.get("avg_yield") if isinstance(yd, dict) else yd
        try:
            avg = float(avg)
        except Exception:
            avg = 0.0
        if avg <= 0.0:
            corrected[dom] = w
        else:
            corrected[dom] = w / avg
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output YAML path (in-repo).")
    ap.add_argument("--split-rel", required=True, help="Relative path under data_root (e.g., data/dedup/... ).")
    ap.add_argument("--tokenizer-var", default="TOKENIZER_JSON", help="Env var name used in manifest for tokenizer.")
    ap.add_argument("--yield-json", default=None, help="Optional yield json to correct weights (domain->avg_yield).")
    args = ap.parse_args()

    base_weights = _normalized_weights(STAGEA_WEIGHTS_PCT)
    if args.yield_json:
        yield_data = _load_yield_json(args.yield_json)
        corrected = _apply_yield_correction(base_weights, yield_data)
        domain_weights = _normalized_weights(corrected)
    else:
        domain_weights = base_weights

    lines = []
    lines.append("seed: 1234")
    lines.append("chars_per_token: 4.0")
    lines.append("")
    lines.append("outputs:")
    lines.append("  raw_dir: data/raw")
    lines.append("  normalized_dir: data/normalized")
    lines.append("  dedup_dir: data/dedup")
    lines.append("  packed_dir: data/packed")
    lines.append("")
    lines.append("dedup:")
    lines.append("  mode: exact")
    lines.append("  scope: per-source")
    lines.append("  simhash_threshold: 3")
    lines.append("  hash_bits: 64")
    lines.append("")
    lines.append("packing:")
    lines.append("  seq_len: 4096")
    lines.append(f"  tokenizer_path: ${{{args.tokenizer_var}}}")
    lines.append("")
    lines.append("domains:")

    for dom, w in domain_weights.items():
        path = f"{args.split_rel}/{dom}.jsonl.zst"
        lines.append(f"  {dom}:")
        lines.append(f"    weight: {w}")
        lines.append("    sources:")
        lines.append("      split:")
        lines.append("        weight: 1.0")
        lines.append("        type: local_dir")
        lines.append("        params:")
        lines.append("          paths:")
        lines.append(f"            - \"{path}\"")
        lines.append("")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Normalized domain weights:", domain_weights)
    print(f"Wrote manifest: {out_path}")


if __name__ == "__main__":
    main()
