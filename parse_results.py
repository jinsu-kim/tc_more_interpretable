import os
import re
import json
import ast
import math
import argparse
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch


# =========================================================
# stdout tee logging
# =========================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def default_log_path(scores_dir):
    os.makedirs(scores_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(scores_dir, f"parse_results_summary_{ts}.log")


# =========================================================
# safe parsing
# =========================================================
def safe_json_load(line):
    try:
        return json.loads(line)
    except Exception:
        return None


def parse_explanation(response: str) -> Optional[str]:
    """
    Strictly parse explainer output for reproduction.

    Accept only responses that contain an explicit [EXPLANATION]: line.
    Degenerate outputs such as repeated punctuation, empty strings, or very short
    strings are rejected instead of being saved as explanations.
    """
    if not response:
        return None

    lines = response.strip().split("\n")
    for line in reversed(lines):
        if "[EXPLANATION]:" not in line:
            continue

        explanation = line.split("[EXPLANATION]:", 1)[1].strip()

        # Reject empty or almost-empty explanations.
        if len(explanation) < 5:
            return None

        # Reject outputs with no alphanumeric content, e.g. "!!!!!!!!".
        if not any(ch.isalnum() for ch in explanation):
            return None

        # Reject obvious degeneration: too many exclamation marks or one repeated
        # non-alphanumeric character after removing spaces.
        compact = explanation.replace(" ", "")
        if explanation.count("!") >= 5:
            return None
        if compact and len(set(compact)) <= 2 and not any(ch.isalnum() for ch in compact):
            return None

        return explanation

    # Do not fall back to the last line. Missing [EXPLANATION]: means invalid.
    return None


def parse_score_list(response: str) -> Optional[List[int]]:
    """Parse a scorer response that should contain exactly one list of 0/1 labels."""
    if not response:
        return None

    match = re.search(r"\[[^\]]+\]", response, flags=re.DOTALL)
    if not match:
        return None

    text = match.group()
    try:
        raw = json.loads(text)
    except Exception:
        try:
            raw = ast.literal_eval(text)
        except Exception:
            return None

    if not isinstance(raw, list):
        return None

    parsed = []
    for x in raw:
        if x in (0, 1):
            parsed.append(int(x))
        elif isinstance(x, str) and x.strip() in ("0", "1"):
            parsed.append(int(x.strip()))
        else:
            return None

    return parsed


# =========================================================
# paper-style score calculation
# =========================================================
def compute_score_dict(predictions: List[int], n_pos: int, n_neg: int):
    """
    Paper-reproduction scoring.

    The request order is assumed to be:
      - first n_pos examples: activating / positive label = 1
      - next  n_neg examples: non-activating / negative label = 0

    Responses with a length different from n_pos + n_neg are rejected.
    """
    expected = n_pos + n_neg
    if len(predictions) != expected:
        return None

    pos = predictions[:n_pos]
    neg = predictions[n_pos:]

    tp = sum(p == 1 for p in pos)
    fn = sum(p == 0 for p in pos)
    fp = sum(p == 1 for p in neg)
    tn = sum(p == 0 for p in neg)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    pos_acc = tp / n_pos if n_pos else 0.0
    neg_acc = tn / n_neg if n_neg else 0.0
    accuracy = (tp + tn) / expected if expected else 0.0
    balanced_accuracy = (pos_acc + neg_acc) / 2

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": expected,
    }


# =========================================================
# filename parsing
# =========================================================
def parse_combo_key(filename: str):
    name = filename.replace("_out.jsonl", "")

    m = re.match(r"^(\w+)_layer(\d+)_(sae|tc)_k(\d+)_explain$", name)
    if m:
        return dict(model=m.group(1), layer=int(m.group(2)),
                    mode=m.group(3), k=int(m.group(4)),
                    task="explain", scorer=None)

    m = re.match(r"^(\w+)_layer(\d+)_(sae|tc)_k(\d+)_(detect|fuzz)_(llama|gpt4o)$", name)
    if m:
        return dict(model=m.group(1), layer=int(m.group(2)),
                    mode=m.group(3), k=int(m.group(4)),
                    task=m.group(5), scorer=m.group(6))

    return None


def parse_feat_idx(custom_id: str):
    m = re.search(r"_feat(\d+)_", custom_id)
    return int(m.group(1)) if m else None


def parse_chunk_idx(custom_id: str):
    m = re.search(r"_chunk(\d+)$", custom_id)
    return int(m.group(1)) if m else None


# =========================================================
# result parsing
# =========================================================
def parse_explain_results(results_path, scores_dir, model, layer, mode, k):
    save_dir = os.path.join(scores_dir, model, f"layer{layer}", f"{mode}_k{k}")
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, "explanations.pt")

    explanations = {}
    if os.path.exists(out_path):
        explanations = torch.load(out_path)

    added = 0
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            obj = safe_json_load(line)
            if obj is None:
                continue

            feat_idx = parse_feat_idx(obj.get("custom_id", ""))
            if feat_idx is None or feat_idx in explanations:
                continue

            if obj.get("error") or not obj.get("response"):
                continue

            explanation = parse_explanation(obj["response"])
            if explanation:
                explanations[feat_idx] = explanation
                added += 1

    torch.save(explanations, out_path)
    print(f"[explain] {model} layer{layer} {mode}_k{k}: total={len(explanations)} added={added}")


def parse_score_results(
    results_path,
    scores_dir,
    model,
    layer,
    mode,
    k,
    task,
    scorer,
    n_score_examples,
    score_chunk_size,
):
    """
    Parse chunked scorer outputs.

    Expected custom_id:
      {combo}_feat{feat_idx}_{task}_{scorer}_chunk{chunk_idx}

    Each chunk contains score_chunk_size examples, except the final chunk may be
    shorter if total examples is not divisible. The full feature-level prediction
    is reconstructed by concatenating chunks in chunk_idx order.
    """
    save_dir = os.path.join(scores_dir, model, f"layer{layer}", f"{mode}_k{k}")
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"{task}_{scorer}.pt")

    scores: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(out_path):
        scores = torch.load(out_path)

    expected_total = n_score_examples * 2
    expected_chunks = math.ceil(expected_total / score_chunk_size)

    chunks: Dict[int, Dict[int, List[int]]] = {}
    bad_parse = 0
    bad_len = 0
    duplicate_chunk = 0

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            obj = safe_json_load(line)
            if obj is None:
                bad_parse += 1
                continue

            custom_id = obj.get("custom_id", "")
            feat_idx = parse_feat_idx(custom_id)
            chunk_idx = parse_chunk_idx(custom_id)

            if feat_idx is None:
                bad_parse += 1
                continue

            # Backward compatibility: if no chunk id exists, treat as old unchunked format.
            if chunk_idx is None:
                chunk_idx = 0

            if obj.get("error") or not obj.get("response"):
                bad_parse += 1
                continue

            preds = parse_score_list(obj["response"])
            if preds is None:
                bad_parse += 1
                continue

            start = chunk_idx * score_chunk_size
            expected_len = min(score_chunk_size, expected_total - start)
            if expected_len <= 0:
                bad_len += 1
                continue

            # For chunked requests, be strict about missing labels; tolerate extra
            # labels by truncating because some scorers repeat or append after the
            # correct list. With chunk_size=5, this should rarely happen.
            if len(preds) < expected_len:
                bad_len += 1
                continue
            preds = preds[:expected_len]

            feat_chunks = chunks.setdefault(feat_idx, {})
            if chunk_idx in feat_chunks:
                duplicate_chunk += 1
                continue
            feat_chunks[chunk_idx] = preds

    added = 0
    incomplete = 0

    for feat_idx, feat_chunks in chunks.items():
        if feat_idx in scores:
            continue

        missing = [i for i in range(expected_chunks) if i not in feat_chunks]
        if missing:
            incomplete += 1
            continue

        preds = []
        for i in range(expected_chunks):
            preds.extend(feat_chunks[i])

        preds = preds[:expected_total]
        score = compute_score_dict(preds, n_score_examples, n_score_examples)
        if score is None:
            bad_len += 1
            continue

        score["n_chunks"] = expected_chunks
        score["score_chunk_size"] = score_chunk_size
        scores[feat_idx] = score
        added += 1

    torch.save(scores, out_path)
    print(
        f"[{task}_{scorer}] {model} layer{layer} {mode}_k{k}: "
        f"total={len(scores)} added={added} bad_parse={bad_parse} "
        f"bad_len={bad_len} incomplete={incomplete} duplicate_chunk={duplicate_chunk}"
    )


# =========================================================
# summary helpers
# =========================================================
def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 else (ys[mid - 1] + ys[mid]) / 2


def _ordered_feature_selection(common, final_n: Optional[int] = 500):
    ordered = sorted(int(x) for x in common)
    if final_n is None or final_n <= 0:
        return ordered
    return ordered[:final_n]


def check_alignment(combo_dir, scorer="llama"):
    exp_path = combo_dir / "explanations.pt"
    det_path = combo_dir / f"detect_{scorer}.pt"
    fuzz_path = combo_dir / f"fuzz_{scorer}.pt"

    if not (exp_path.exists() and det_path.exists() and fuzz_path.exists()):
        return

    exp = torch.load(exp_path)
    det = torch.load(det_path)
    fuz = torch.load(fuzz_path)

    common_all = set(exp.keys()) & set(det.keys()) & set(fuz.keys())
    common_scores = set(det.keys()) & set(fuz.keys())

    if len(common_all) < 0.8 * len(common_scores):
        print(
            f"[WARN] low alignment: {combo_dir} scorer={scorer} "
            f"exp∩det∩fuzz={len(common_all)} det∩fuzz={len(common_scores)}"
        )


def _score_combo(combo_dir, scorer="llama", final_n: Optional[int] = 500, metric: str = "accuracy"):
    """
    Read detect/fuzz scores for one combo and return per-feature final scores.
    Final feature score = mean(detection metric, fuzzing metric).
    metric can be: accuracy, balanced_accuracy, or f1.
    """
    detect = combo_dir / f"detect_{scorer}.pt"
    fuzz = combo_dir / f"fuzz_{scorer}.pt"

    if not (detect.exists() and fuzz.exists()):
        return []

    check_alignment(combo_dir, scorer=scorer)

    det = torch.load(detect)
    fuz = torch.load(fuzz)

    common = set(det.keys()) & set(fuz.keys())
    selected = _ordered_feature_selection(common, final_n=final_n)

    if final_n is not None and final_n > 0 and len(selected) < final_n:
        print(f"[WARN] only {len(selected)}/{final_n} complete features: {combo_dir} scorer={scorer}")

    scores = []
    for feat_idx in selected:
        if metric not in det[feat_idx] or metric not in fuz[feat_idx]:
            continue
        scores.append((det[feat_idx][metric] + fuz[feat_idx][metric]) / 2)

    return scores

def collect_grouped_scores(scores_dir, scorer="llama", final_n: Optional[int] = 500, metric: str = "accuracy"):
    grouped = {}

    root = Path(scores_dir)
    if not root.exists():
        return grouped

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for layer_dir in sorted(model_dir.glob("layer*")):
            if not layer_dir.is_dir():
                continue
            m_layer = re.match(r"^layer(\d+)$", layer_dir.name)
            if not m_layer:
                continue
            layer = int(m_layer.group(1))

            for combo_dir in sorted(layer_dir.glob("*_k*")):
                if not combo_dir.is_dir():
                    continue
                m_combo = re.match(r"^(sae|tc)_k(\d+)$", combo_dir.name)
                if not m_combo:
                    continue

                mode = m_combo.group(1)
                k = int(m_combo.group(2))
                scores = _score_combo(combo_dir, scorer=scorer, final_n=final_n, metric=metric)
                if scores:
                    grouped[(model, layer, k, mode)] = scores

    return grouped


def print_layerwise_summary(grouped, metric_label="score"):
    print("\n==== LAYERWISE SUMMARY ====")
    models = sorted({key[0] for key in grouped.keys()})

    for model in models:
        print(f"\n==== {model} ====")
        layers = sorted({layer for (m, layer, k, mode) in grouped.keys() if m == model})
        ks = sorted({k for (m, layer, k, mode) in grouped.keys() if m == model})

        for k in ks:
            print(f"\n-- k{k} --")
            print(f"layer\tSAE_mean_{metric_label}\tTC_mean_{metric_label}\tdiff(TC-SAE)\tSAE_n\tTC_n")

            for layer in layers:
                sae_scores = grouped.get((model, layer, k, "sae"), [])
                tc_scores = grouped.get((model, layer, k, "tc"), [])
                if not sae_scores and not tc_scores:
                    continue

                sae_mean = _mean(sae_scores)
                tc_mean = _mean(tc_scores)
                sae_str = f"{sae_mean:.4f}" if sae_mean is not None else "NA"
                tc_str = f"{tc_mean:.4f}" if tc_mean is not None else "NA"
                diff_str = f"{tc_mean - sae_mean:+.4f}" if sae_mean is not None and tc_mean is not None else "NA"

                print(
                    f"layer{layer}\t{sae_str}\t{tc_str}\t{diff_str}\t"
                    f"{len(sae_scores)}\t{len(tc_scores)}"
                )


def print_overall_summary(grouped):
    print("\n==== OVERALL SUMMARY ====")
    models = sorted({key[0] for key in grouped.keys()})

    for model in models:
        sae_all = []
        tc_all = []
        for (m, layer, k, mode), scores in grouped.items():
            if m != model:
                continue
            if mode == "sae":
                sae_all.extend(scores)
            elif mode == "tc":
                tc_all.extend(scores)

        if sae_all and tc_all:
            sae_mean = _mean(sae_all)
            tc_mean = _mean(tc_all)
            print(f"\n{model}")
            print(f"SAE: {sae_mean:.4f} (n={len(sae_all)})")
            print(f"TC : {tc_mean:.4f} (n={len(tc_all)})")
            print(f"diff: {tc_mean - sae_mean:+.4f}")


def print_task_breakdown(scores_dir, scorer="llama", final_n: Optional[int] = 500, metric: str = "accuracy"):
    print(f"\n==== TASK BREAKDOWN: detect / fuzz ({scorer}, {metric}) ====")
    rows = []
    root = Path(scores_dir)
    if not root.exists():
        return

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for layer_dir in sorted(model_dir.glob("layer*")):
            m_layer = re.match(r"^layer(\d+)$", layer_dir.name)
            if not m_layer:
                continue
            layer = int(m_layer.group(1))

            for combo_dir in sorted(layer_dir.glob("*_k*")):
                m_combo = re.match(r"^(sae|tc)_k(\d+)$", combo_dir.name)
                if not m_combo:
                    continue
                mode = m_combo.group(1)
                k = int(m_combo.group(2))

                detect_path = combo_dir / f"detect_{scorer}.pt"
                fuzz_path = combo_dir / f"fuzz_{scorer}.pt"
                if not (detect_path.exists() and fuzz_path.exists()):
                    continue

                det = torch.load(detect_path)
                fuz = torch.load(fuzz_path)
                selected = _ordered_feature_selection(set(det.keys()) & set(fuz.keys()), final_n=final_n)
                if not selected:
                    continue

                det_scores = [det[f][metric] for f in selected if metric in det[f]]
                fuzz_scores = [fuz[f][metric] for f in selected if metric in fuz[f]]
                rows.append((model, layer, k, mode, _mean(det_scores), _mean(fuzz_scores), len(selected)))

    if not rows:
        return

    current = None
    for model, layer, k, mode, det_mean, fuzz_mean, n in rows:
        header = (model, k)
        if header != current:
            current = header
            print(f"\n==== {model} k{k} ====")
            print(f"layer\tmode\tdetect_{metric}\tfuzz_{metric}\tn")
        print(f"layer{layer}\t{mode}\t{det_mean:.4f}\t{fuzz_mean:.4f}\t{n}")

def print_summary(scores_dir, scorer="llama", final_n: Optional[int] = 500, metric: str = "accuracy"):
    grouped = collect_grouped_scores(scores_dir, scorer=scorer, final_n=final_n, metric=metric)

    if not grouped:
        print("\n==== SUMMARY ====")
        print(f"No complete detect/fuzz score files found for scorer={scorer}.")
        return

    print(f"\n[SCORER] {scorer}")
    print(f"[FINAL_N] {final_n if final_n is not None and final_n > 0 else 'all complete features'}")
    print(f"[METRIC] mean of detection {metric} and fuzzing {metric}")

    print_layerwise_summary(grouped, metric_label=metric)
    print_overall_summary(grouped)
    print_task_breakdown(scores_dir, scorer=scorer, final_n=final_n, metric=metric)


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--scores_dir", default="./scores")
    parser.add_argument("--n_score_examples", type=int, default=50)
    parser.add_argument("--score_chunk_size", type=int, default=5,
                        help="Number of examples per scorer chunk. Use 5 for Delphi-style chunked scoring.")
    parser.add_argument("--scorer", choices=["llama", "gpt4o"], default="llama",
                        help="Which scorer files to summarize: detect_<scorer>.pt and fuzz_<scorer>.pt")
    parser.add_argument("--metric", choices=["accuracy", "balanced_accuracy", "f1", "all"], default="accuracy",
                        help="Metric used for the final table. Use all to print accuracy, balanced_accuracy, and f1 in one run.")
    parser.add_argument("--final_n", type=int, default=500,
                        help="Number of complete features per model/layer/mode/k. Use <=0 for all complete features.")
    parser.add_argument("--log_path", default=None,
                        help="Path to save terminal summary output. Default: scores_dir/parse_results_summary_<timestamp>.log")
    args = parser.parse_args()

    result_files = sorted(Path(args.results_dir).glob("*_out.jsonl"))
    final_n = args.final_n if args.final_n > 0 else None

    for path in result_files:
        info = parse_combo_key(path.name)
        if info is None:
            continue

        if info["task"] == "explain":
            parse_explain_results(str(path), args.scores_dir, info["model"], info["layer"], info["mode"], info["k"])
        else:
            parse_score_results(str(path), args.scores_dir,
                                info["model"], info["layer"], info["mode"], info["k"],
                                info["task"], info["scorer"],
                                args.n_score_examples,
                                args.score_chunk_size)

    log_path = args.log_path or default_log_path(args.scores_dir)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    metrics = ["accuracy", "balanced_accuracy", "f1"] if args.metric == "all" else [args.metric]

    with open(log_path, "w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with redirect_stdout(tee):
            for idx, metric in enumerate(metrics):
                if len(metrics) > 1:
                    if idx > 0:
                        print("\n" + "=" * 80)
                    print(f"\n######## METRIC: {metric} ########")
                print_summary(args.scores_dir, scorer=args.scorer, final_n=final_n, metric=metric)
            print(f"\n[LOG SAVED] {log_path}")

if __name__ == "__main__":
    main()
