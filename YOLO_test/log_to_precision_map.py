#!/usr/bin/env python3
"""Convert benchmark run logs into per-layer precision-map JSON files.

The detection benchmark (yolo_test.py) prints one line per LoF layer of the
form:

    Layer: model.1.conv  |  Format: ({('binary8p4', 'binary8p4', 'binary8p4', \
'binary29p21'): 100.0}, {(32, 1, 144): {('binary8p4', 'binary8p4', 'binary8p4', \
'binary29p21'): 1}})

where the 4-tuple is (activation, weight, bias, accumulator) in LoFloat
`binary{total}p{mantissa}` notation.  This script scrapes those lines and emits
the same precision-config JSON that sensitivity_search.save_precision_config
writes, so an old run can be replayed with load_precision_config without
re-running the search.

A single log may contain several runs (multiple models / datasets / accuracy
targets).  Each contiguous block of Layer lines becomes its own config; a new
block is detected when a layer name repeats.  Best-effort run metadata
(model, dataset, accuracy target, FP-kept first/last layers) is scraped from
the surrounding lines.

Usage:
    python log_to_precision_map.py LOG.txt [LOG.txt ...]
    python log_to_precision_map.py logs/*.txt -o precision_maps/
    python log_to_precision_map.py run.txt --stdout        # print, don't write

Output: for a log with a single run, <log>.precision.json (or <name> in -o dir).
For multiple runs, a numbered/keyed suffix is appended, e.g.
<log>.yolov8n_coco_t0.01.precision.json.
"""
import argparse
import ast
import json
import os
import re
import sys

PRECISION_CONFIG_VERSION = 1

# "  Layer: <name>  |  Format: <python-literal>"
_LAYER_RE = re.compile(r"^\s*Layer:\s*(?P<name>\S+)\s*\|\s*Format:\s*(?P<fmt>.+?)\s*$")
_FMT_RE = re.compile(r"^binary(\d+)p(\d+)$")

# --- best-effort metadata markers ---
_TARGET_RE = re.compile(r"accuracy[_ ]target\s*[=:]\s*([0-9]*\.?[0-9]+)")
_MODELDS_RE = re.compile(r"\[([\w.\-]+)/(coco|pascal)\]")
_SKIP_RE = re.compile(r"(?:First/last layers|keeping in full precision)[^\[]*(\[[^\]]*\])")


def parse_format_string(fmt):
    """'binary8p4' -> dict(mantissa=4, exponent=3, total=8, format='binary8p4')."""
    m = _FMT_RE.match(fmt)
    if not m:
        raise ValueError(f"not a LoFloat format string: {fmt!r}")
    total, mant = int(m.group(1)), int(m.group(2))
    return {
        "mantissa": mant,
        "exponent": total - mant - 1,
        "total": total,
        "format": fmt,
    }


def _dominant_format_tuple(fmt_literal):
    """Parse the Format field and return its (act, weight, bias, accum) tuple.

    The field is a Python literal `(formats_flops, gemm_stats)` where
    formats_flops maps a format-tuple to a flops share.  A single module yields
    exactly one tuple; if several appear, the one with the largest share wins."""
    obj = ast.literal_eval(fmt_literal)
    formats_flops = obj[0] if isinstance(obj, tuple) else obj
    if not isinstance(formats_flops, dict) or not formats_flops:
        raise ValueError("no format tuple in Format field")
    best = max(formats_flops.items(), key=lambda kv: kv[1])[0]
    if not (isinstance(best, tuple) and len(best) == 4):
        raise ValueError(f"unexpected format tuple: {best!r}")
    return best


def _layer_entry(fmt_tuple):
    act, weight, bias, accum = fmt_tuple
    total, mant = int(_FMT_RE.match(accum).group(1)), int(_FMT_RE.match(accum).group(2))
    # record_formats prints accum as name_type(9 + mant_bits, 1 + mant_bits),
    # so accum mantissa field == 1 + accum_mant_bits.
    accum_mant_bits = mant - 1
    return {
        "activation": parse_format_string(act),
        "weight": parse_format_string(weight),
        "bias": parse_format_string(bias),
        "accum": {"mant_bits": accum_mant_bits, "format": accum},
    }


def _scrape_meta(lines):
    """Best-effort run metadata from a slice of surrounding log lines.
    Later occurrences win, so a slice ending at the run keeps the nearest
    accuracy_target / skip list, and model/dataset markers (printed after the
    layer block) are picked up when the slice extends past it."""
    meta = {}
    for line in lines:
        t = _TARGET_RE.search(line)
        if t:
            meta["accuracy_target"] = float(t.group(1))
        md = _MODELDS_RE.search(line)
        if md:
            meta["model"], meta["dataset"] = md.group(1), md.group(2)
        sk = _SKIP_RE.search(line)
        if sk:
            try:
                meta["skip_layers"] = list(ast.literal_eval(sk.group(1)))
            except (ValueError, SyntaxError):
                pass
    return meta


def parse_log(path):
    """Return a list of run dicts: {'meta': {...}, 'layers': {name: entry}}.

    A run is a block of Layer lines; blocks may be interleaved with other
    output (Weights/stats lines), so a new block is detected only when a layer
    name repeats.  Metadata is scraped from the lines spanning each block plus
    the gap after it (model/dataset are printed after the block)."""
    with open(path, errors="replace") as f:
        lines = f.readlines()

    # Pass 1: collect (line_index, name, entry) for every Layer line.
    hits = []
    for idx, line in enumerate(lines):
        m = _LAYER_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        try:
            entry = _layer_entry(_dominant_format_tuple(m.group("fmt")))
        except (ValueError, SyntaxError) as e:
            print(f"  [warn] {os.path.basename(path)}: skipping layer "
                  f"{name!r}: {e}", file=sys.stderr)
            continue
        hits.append((idx, name, entry))

    # Pass 2: split into blocks whenever a name repeats within the block.
    blocks, cur = [], {}
    for idx, name, entry in hits:
        if name in cur:
            blocks.append(cur)
            cur = {}
        cur[name] = (idx, entry)
    if cur:
        blocks.append(cur)

    # Pass 3: attach metadata scraped from each block's line span + trailing gap.
    runs = []
    for b, block in enumerate(blocks):
        idxs = [i for i, _ in block.values()]
        start = min(idxs)
        # Extend to the start of the next block so post-block markers (model/
        # dataset) are captured; clamp to end of file for the last block.
        if b + 1 < len(blocks):
            end = min(i for i, _ in blocks[b + 1].values())
        else:
            end = len(lines)
        # Also look a little before the block for the accuracy_target header.
        prev_end = max((i for i, _ in blocks[b - 1].values()), default=0) if b else 0
        meta = _scrape_meta(lines[prev_end:end])
        layers = {name: entry for name, (_, entry) in block.items()}
        runs.append({"meta": meta, "layers": layers})
    return runs


def _run_suffix(meta, index, n_runs):
    if n_runs == 1:
        return ""
    parts = []
    if meta.get("model"):
        parts.append(str(meta["model"]))
    if meta.get("dataset"):
        parts.append(str(meta["dataset"]))
    if meta.get("accuracy_target") is not None:
        parts.append(f"t{meta['accuracy_target']}")
    parts.append(f"run{index}")
    return "." + "_".join(parts)


def convert(path, outdir=None, to_stdout=False):
    runs = parse_log(path)
    if not runs:
        print(f"[skip] no 'Layer: ... | Format:' lines in {path}", file=sys.stderr)
        return []

    base = os.path.splitext(os.path.basename(path))[0]
    written = []
    for i, run in enumerate(runs):
        run["meta"]["source_log"] = os.path.abspath(path)
        config = {
            "version": PRECISION_CONFIG_VERSION,
            "meta": run["meta"],
            "layers": run["layers"],
        }
        if to_stdout:
            print(json.dumps(config, indent=2))
            continue
        suffix = _run_suffix(run["meta"], i, len(runs))
        out_name = f"{base}{suffix}.precision.json"
        out_path = os.path.join(outdir or os.path.dirname(path) or ".", out_name)
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[ok] {path}: run {i} ({len(run['layers'])} layers) -> {out_path}")
        written.append(out_path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="run log .txt file(s)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="directory for output JSON (default: alongside each log)")
    ap.add_argument("--stdout", action="store_true",
                    help="print JSON to stdout instead of writing files")
    args = ap.parse_args(argv)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    total = 0
    for path in args.logs:
        total += len(convert(path, outdir=args.outdir, to_stdout=args.stdout))
    if not args.stdout:
        print(f"\nWrote {total} precision map(s).")


if __name__ == "__main__":
    main()
