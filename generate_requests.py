
import os
import json
import random
import argparse
from typing import Optional, List, Dict
from pathlib import Path

import torch
from transformers import AutoTokenizer


# =========================================================
# =========================================================
MODEL_CONFIGS = {
    "gpt2s": {
        "hf_name": "gpt2",
        "n_layers": 12,
        "enabled": True,
    },
    "pythia160m": {
        "hf_name": "EleutherAI/pythia-160m",
        "n_layers": 12,
        "enabled": True,
    },
    "pythia410m": {
        "hf_name": "EleutherAI/pythia-410m",
        "n_layers": 24,
        "enabled": True,
    },
}

MODES = ["sae", "tc"]
TOPKS = [32, 64]

NOVITA_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENAI_MODEL = "gpt-4o-mini"


# =========================================================
# =========================================================

# --- Explainer ---
EXPLAINER_SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text examples and provide an explanation that captures the common latent pattern.

Guidelines:

You will be given a list of text examples in which selected tokens are marked between delimiters like <<this>>. If a sequence of consecutive tokens is important, the entire sequence will be contained between delimiters <<just like this>>. The activation value for the marked token is listed after each example in parentheses.

- Produce a concise final description of the latent pattern common to the marked tokens.
- If some examples are uninformative, you do not need to mention them.
- Do not merely list example tokens; summarize the pattern they share.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations.
- Keep your explanation short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:
"""

EXPLAINER_FEW_SHOT = [
    {
        "role": "user",
        "content": "\nExample 1:  and he was <<over the moon>> to find\nExample 2:  we'll be laughing <<till the cows come home>>! Pro\nExample 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd\n"
    },
    {
        "role": "assistant",
        "content": "\n[EXPLANATION]: Common idioms in text conveying positive sentiment.\n"
    },
    {
        "role": "user",
        "content": "\nExample 1:  a river is wide but the ocean is wid<<er>>. The ocean\nExample 2:  every year you get tall<<er>>,\" she\nExample 3:  the hole was small<<er>> but deep<<er>> than the\n"
    },
    {
        "role": "assistant",
        "content": "\n[EXPLANATION]: The token \"er\" at the end of a comparative adjective describing size.\n"
    },
]

# --- Detection scorer ---
DETECTION_SYSTEM = """You are an intelligent and meticulous linguistics researcher.

You will be given a latent explanation, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Your task is to determine which examples contain the described latent.

For each example in order, return 1 if the example contains the latent and 0 if it does not. You must return your response as a valid Python list of integers. Do not return anything except the Python list.
"""

DETECTION_FEW_SHOT = [
    {
        "role": "user",
        "content": 'Latent explanation: Words related to American football positions, specifically the tight end position.\n\nTest examples:\n\nExample 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his boss\nExample 1: names of months used in The Lord of the Rings:ĊĊ\nExample 2: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the\nExample 3: shown, is generally not eligible for ads. For example, videos about recent tragedies,\nExample 4: line, with the left side namely tackle Byron Bell at tackle and guard Amini\n'
    },
    {"role": "assistant", "content": "[1,0,0,0,1]"},
    {
        "role": "user",
        "content": 'Latent explanation: The word "guys" in the phrase "you guys".\n\nTest examples:\n\nExample 0: enact an individual health insurance mandate?, Pelosi\'s response was to dismiss both\nExample 1: birth control access but I assure you women in Kentucky aren\'t laughing as they struggle\nExample 2: du Soleil Fall Protection Program with construction requirements that do not apply to theater settings\nExample 3: distasteful. Amidst the slime lurk bits of Schadenfreude\nExample 4: the I want to remind you all that 10 days ago (director Massimil\n'
    },
    {"role": "assistant", "content": "[0,0,0,0,0]"},
    {
        "role": "user",
        "content": 'Latent explanation: "of" before words that start with a capital letter.\n\nTest examples:\n\nExample 0: climate, Tomblin\'s Chief of Staff Charlie Lorensen said.\nExample 1: no wonderworking relics, no true Body and Blood of Christ, no true Baptism\nExample 2: Deborah Sathe, Head of Talent Development and Production at Film London,\nExample 3: It has been devised by Director of Public Prosecutions (DPP)\nExample 4: and fair investigation not even include the Director of Athletics?\n'
    },
    {"role": "assistant", "content": "[1,1,1,1,1]"},
]

# --- Fuzzing scorer ---
FUZZING_SYSTEM = """You are an intelligent and meticulous linguistics researcher.

You will be given a latent explanation, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. In each example, portions of the text are marked between << and >>. Some marked tokens may correctly represent the latent, while others may be mislabeled.

Your task is to determine whether every marked token in each example correctly represents the latent.

For each example in order, return 1 if all marked tokens correctly represent the latent and 0 otherwise. You must return your response as a valid Python list of integers. Do not return anything except the Python list.
"""

FUZZING_FEW_SHOT = [
    {
        "role": "user",
        "content": 'Latent explanation: Words related to American football positions, specifically the tight end position.\n\nTest examples:\n\nExample 0:<|endoftext|>Getty ImagesĊĊPatriots<< tight end>> Rob Gronkowski had his boss\nExample 1: posted You should know this<< about>> offensive line coaches: they are large, demanding<< men>>\nExample 2: Media Day 2015ĊĊLSU<< defensive>> end Isaiah Washington (94) speaks<< to the>>\nExample 3:<< running backs>>," he said. .. Defensive<< end>> Carroll Phillips is improving\nExample 4:<< line>>, with the left side namely<< tackle>> Byron Bell at<< tackle>> and<< guard>> Amini\n'
    },
    {"role": "assistant", "content": "[1,0,0,1,1]"},
    {
        "role": "user",
        "content": 'Latent explanation: The word "guys" in the phrase "you guys".\n\nTest examples:\n\nExample 0: if you are<< comfortable>> with it. You<< guys>> support me in many other ways\nExample 1: birth control access but I assure you<< women>> in Kentucky aren\'t laughing\nExample 2: s gig! I hope you guys<< LOVE>> her, and<< please>> be nice,\nExample 3: American, told<< Hannity>> that you<< guys>> are playing the race card.\nExample 4:<< the>> I want to<< remind>> you all that 10 days ago\n'
    },
    {"role": "assistant", "content": "[0,0,0,0,0]"},
    {
        "role": "user",
        "content": 'Latent explanation: "of" before words that start with a capital letter.\n\nTest examples:\n\nExample 0: climate, Tomblin\'s Chief<< of>> Staff Charlie Lorensen said.\nExample 1: no wonderworking relics, no true Body and Blood<< of>> Christ, no true Baptism\nExample 2: Deborah Sathe, Head<< of>> Talent Development and Production at Film London,\nExample 3: It has been devised by Director<< of>> Public Prosecutions (DPP)\nExample 4: and fair investigation not even include the Director<< of>> Athletics?\n'
    },
    {"role": "assistant", "content": "[1,1,1,1,1]"},
]


# =========================================================
# =========================================================
_tokenizer_cache: Dict[str, AutoTokenizer] = {}

def get_tokenizer(hf_name: str) -> AutoTokenizer:
    if hf_name not in _tokenizer_cache:
        tok = AutoTokenizer.from_pretrained(hf_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _tokenizer_cache[hf_name] = tok
    return _tokenizer_cache[hf_name]


# =========================================================
# =========================================================
def decode_context(token_ids: torch.Tensor, tokenizer: AutoTokenizer) -> str:
    ids = token_ids.tolist()
    # remove pad tokens
    ids = [i for i in ids if i != tokenizer.pad_token_id]
    return tokenizer.decode(ids, skip_special_tokens=False)


def decode_context_with_highlight(
    token_ids: torch.Tensor,
    center_idx: int,
    tokenizer: AutoTokenizer,
    ctx_len: int = 32,
) -> str:

    ids = token_ids.tolist()
    tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]

    # highlight the center token
    if 0 <= center_idx < len(tokens):
        tokens[center_idx] = f"<<{tokens[center_idx]}>>"

    return "".join(tokens)


# =========================================================
# =========================================================
def sample_features(
    alive_features: List[int],
    n_features: int,
    seed: int = 42,
) -> List[int]:
    """Randomly sampling n_features from alive features"""
    random.seed(seed)
    if len(alive_features) <= n_features:
        return alive_features
    return random.sample(alive_features, n_features)


def sample_examples_stratified(
    activations: torch.Tensor,
    contexts: torch.Tensor,
    n_examples: int,
    n_quantiles: int = 10,
    exclude_indices: Optional[List[int]] = None,
) -> tuple:
    """
    Activation-value quantile sampling for auto-interpretability.

    Returns:
        sampled_activations, sampled_contexts, sampled_original_indices

    Paper-style policy:
    - explanation: 40 activating examples = 10 quantiles x 4
    - scoring: 50 activating examples = 10 quantiles x 5

    If exclude_indices is provided, those cache positions are removed first so
    explanation and scoring examples do not overlap.
    """
    n = len(activations)
    if n == 0:
        return activations, contexts, []

    if exclude_indices:
        exclude_set = set(int(i) for i in exclude_indices)
        valid_mask = torch.tensor(
            [i not in exclude_set for i in range(n)],
            dtype=torch.bool,
            device=activations.device,
        )
        if valid_mask.sum().item() == 0:
            valid_mask = torch.ones(n, dtype=torch.bool, device=activations.device)
    else:
        valid_mask = torch.ones(n, dtype=torch.bool, device=activations.device)

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    acts_valid = activations[valid_indices].float()
    ctxs_valid = contexts[valid_indices.cpu()]

    if acts_valid.numel() == 0:
        return activations[:0], contexts[:0], []

    per_q = max(1, n_examples // n_quantiles)
    qs = torch.linspace(0, 1, n_quantiles + 1, device=acts_valid.device)
    q_bounds = torch.quantile(acts_valid, qs)

    selected_chunks = []
    selected_set = set()

    for q in range(n_quantiles):
        low = q_bounds[q]
        high = q_bounds[q + 1]

        if q == n_quantiles - 1:
            mask = (acts_valid >= low) & (acts_valid <= high)
        else:
            mask = (acts_valid >= low) & (acts_valid < high)

        q_idx = mask.nonzero(as_tuple=True)[0]
        if q_idx.numel() == 0:
            continue

        # Avoid duplicate positions when quantile boundaries collapse.
        q_idx_list = [int(i) for i in q_idx.tolist() if int(i) not in selected_set]
        if not q_idx_list:
            continue

        q_idx = torch.tensor(q_idx_list, dtype=torch.long, device=acts_valid.device)
        k = min(per_q, q_idx.numel())
        chosen = q_idx[torch.randperm(q_idx.numel(), device=acts_valid.device)[:k]]

        for i in chosen.tolist():
            selected_set.add(int(i))
        selected_chunks.append(chosen)

    if selected_chunks:
        selected = torch.cat(selected_chunks)
    else:
        selected = torch.empty(0, dtype=torch.long, device=acts_valid.device)

    # Fill shortages from the remaining valid examples.
    if selected.numel() < n_examples:
        remaining = [i for i in range(acts_valid.numel()) if i not in selected_set]
        if remaining:
            remaining = torch.tensor(remaining, dtype=torch.long, device=acts_valid.device)
            k = min(n_examples - selected.numel(), remaining.numel())
            extra = remaining[torch.randperm(remaining.numel(), device=acts_valid.device)[:k]]
            selected = torch.cat([selected, extra])

    selected = selected[:n_examples]
    original_indices = valid_indices[selected].cpu().tolist()

    return acts_valid[selected].cpu(), ctxs_valid[selected.cpu()], original_indices

def sample_non_activating(
    all_contexts: Dict[int, torch.Tensor],
    feat_idx: int,
    n: int,
    tokenizer: AutoTokenizer,
    seed: int = 42,
) -> List[str]:
    """
    Approximate negative fallback.

    This samples contexts from other features and does NOT verify that the
    target feature is inactive. For paper-reproduction runs, prefer strict
    negatives from non_activating_contexts.pt and keep allow_approx_negatives=False.
    """
    random.seed(seed + feat_idx)
    other_feats = [k for k in all_contexts.keys() if k != feat_idx]
    if not other_feats:
        return []

    results = []
    attempts = 0
    while len(results) < n and attempts < n * 10:
        f = random.choice(other_feats)
        ctxs = all_contexts[f]
        idx = random.randint(0, len(ctxs) - 1)
        text = decode_context(ctxs[idx], tokenizer)
        results.append(text)
        attempts += 1

    return results[:n]


def sample_non_activating_from_pool(
    non_activating_contexts: Dict[int, torch.Tensor],
    feat_idx: int,
    n: int,
    tokenizer: AutoTokenizer,
    seed: int = 42,
) -> List[str]:
    """
    Strict negative examples.
    non_activating_contexts[feat_idx] must contain contexts where the target
    feature activation is below the chosen threshold.
    """
    random.seed(seed + feat_idx)

    if feat_idx not in non_activating_contexts:
        return []

    ctxs = non_activating_contexts[feat_idx]
    if len(ctxs) == 0:
        return []

    indices = list(range(len(ctxs)))
    random.shuffle(indices)
    indices = indices[:min(n, len(indices))]

    return [decode_context(ctxs[i], tokenizer) for i in indices]


# =========================================================
# =========================================================
def clean_explanation(explanation: str) -> str:
    """Remove '[EXPLANATION]: ' prefix"""
    explanation = str(explanation).strip()
    if "[EXPLANATION]:" in explanation:
        explanation = explanation.split("[EXPLANATION]:", 1)[1].strip()
    return explanation


# =========================================================
# Explainer
# =========================================================
def build_explainer_user_content(
    contexts: torch.Tensor,
    activations: torch.Tensor,
    tokenizer: AutoTokenizer,
    ctx_len: int = 32,
) -> str:
    """
    Construct the explainer user message using 40 examples
    (10 quantiles × 4 examples per quantile).
    Highlight activated tokens with <<...>>.
    """
    ctx_half = ctx_len // 2
    lines = []

    for i, (ctx, act) in enumerate(zip(contexts, activations)):
        text = decode_context_with_highlight(ctx, ctx_half, tokenizer, ctx_len)
        center_token = tokenizer.decode([ctx[ctx_half].item()], skip_special_tokens=False)
        act_val = round(float(act.item()), 3)
        lines.append(
            f'Example {i+1}: {text} '
            f'(Activations: "{center_token}": {act_val})'
        )

    return "\n".join(lines)


def generate_explain_requests(
    cache_dir: str,
    requests_dir: str,
    model_short: str,
    hf_name: str,
    layer_idx: int,
    mode: str,
    k: int,
    n_features: int = 500,
    n_explain_examples: int = 40,
    seed: int = 42,
) -> str:
    """
    """
    combo_key = f"{model_short}_layer{layer_idx}_{mode}_k{k}"
    cache_path = os.path.join(cache_dir, model_short, f"layer{layer_idx}", f"{mode}_k{k}")

    # Load Cache
    alive_path = os.path.join(cache_path, "alive_features.pt")
    act_path = os.path.join(cache_path, "activations.pt")
    ctx_path = os.path.join(cache_path, "token_contexts.pt")

    if not os.path.exists(alive_path):
        print(f"  [skip] No cache: {cache_path}")
        return None

    alive_features = torch.load(alive_path)
    alive_features = [int(x) for x in alive_features]
    activations_dict = torch.load(act_path)
    contexts_dict = torch.load(ctx_path)
    tokenizer = get_tokenizer(hf_name)

    sampled_feats = sample_features(alive_features, n_features, seed=seed)

    os.makedirs(requests_dir, exist_ok=True)
    out_path = os.path.join(requests_dir, f"{combo_key}_explain.jsonl")

    if os.path.exists(out_path):
        print(f"  [skip] Already exists: {out_path}")
        return out_path

    written = 0
    explain_indices: Dict[int, List[int]] = {}

    with open(out_path, "w") as f:
        for feat_idx in sampled_feats:
            feat_idx = int(feat_idx)
            if feat_idx not in activations_dict:
                continue

            acts = activations_dict[feat_idx]
            ctxs = contexts_dict[feat_idx]

            acts_s, ctxs_s, used_indices = sample_examples_stratified(
                acts, ctxs, n_explain_examples, n_quantiles=10
            )
            if len(ctxs_s) < n_explain_examples:
                print(
                    f"  [warn] feat {feat_idx}: insufficient explain positives "
                    f"({len(ctxs_s)}/{n_explain_examples}) → skip"
                )
                continue
            explain_indices[feat_idx] = used_indices

            user_content = build_explainer_user_content(ctxs_s, acts_s, tokenizer)

            messages = [
                {"role": "system", "content": EXPLAINER_SYSTEM},
                *EXPLAINER_FEW_SHOT,
                {"role": "user", "content": user_content},
            ]

            custom_id = f"{combo_key}_feat{feat_idx}_explain"
            record = {
                "custom_id": custom_id,
                "model": NOVITA_MODEL,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    explain_indices_path = os.path.join(requests_dir, f"{combo_key}_explain_indices.pt")
    torch.save(explain_indices, explain_indices_path)

    print(f"  Created explain requests: {written} -> {out_path}")
    return out_path


# =========================================================
# Scorer
# =========================================================
def build_scorer_examples_str(contexts: List[str]) -> str:
    lines = []
    for i, ctx in enumerate(contexts):
        lines.append(f"Example {i}: {ctx}")
    return "\n".join(lines)


def add_random_highlight_to_text(text: str, rng: random.Random) -> str:
    """
    Helper for fuzzing negatives.

    Non-activating contexts do not contain truly active tokens, so this function
    marks an arbitrary non-empty word with <<...>> to create a mislabeled example
    that the scorer should classify as 0.
    """
    parts = text.split(" ")
    candidate_positions = [i for i, part in enumerate(parts) if part.strip()]
    if not candidate_positions:
        return f"<<{text}>>" if text else "<< >>"

    pos = rng.choice(candidate_positions)
    parts[pos] = f"<<{parts[pos]}>>"
    return " ".join(parts)


def generate_score_requests(
    cache_dir: str,
    requests_dir: str,
    scores_dir: str,
    model_short: str,
    hf_name: str,
    layer_idx: int,
    mode: str,
    k: int,
    n_score_examples: int = 50,
    use_gpt4o: bool = False,
    seed: int = 42,
    allow_approx_negatives: bool = False,
    score_chunk_size: int = 5,
) -> List[str]:

    combo_key = f"{model_short}_layer{layer_idx}_{mode}_k{k}"
    cache_path = os.path.join(cache_dir, model_short, f"layer{layer_idx}", f"{mode}_k{k}")
    scores_path = os.path.join(scores_dir, model_short, f"layer{layer_idx}", f"{mode}_k{k}")

    # Check explanations.pt
    explain_pt = os.path.join(scores_path, "explanations.pt")
    if not os.path.exists(explain_pt):
        print(f"  [skip] explanations.pt not found: {explain_pt}")
        return []

    explanations = torch.load(explain_pt)  # {feat_idx: str}

    # Load Cache
    activations_dict = torch.load(os.path.join(cache_path, "activations.pt"))
    contexts_dict = torch.load(os.path.join(cache_path, "token_contexts.pt"))
    tokenizer = get_tokenizer(hf_name)

    # Strict negative pool for paper-style reproduction.
    non_ctx_path = os.path.join(cache_path, "non_activating_contexts.pt")
    non_activating_contexts = None
    if os.path.exists(non_ctx_path):
        non_activating_contexts = torch.load(non_ctx_path)
        print(f"  [info] Loaded strict non-activating contexts: {non_ctx_path}")
    elif allow_approx_negatives:
        print("  [warn] non_activating_contexts.pt not found: using approximate negatives")
    else:
        raise FileNotFoundError(
            f"Strict negative cache not found: {non_ctx_path}. "
            f"non_activating_contexts.pt is required for paper-style score requests. "
            f"Use --allow_approx_negatives to enable approximate fallback negatives."
        )

    # Explanation/score split.
    explain_indices_path = os.path.join(requests_dir, f"{combo_key}_explain_indices.pt")
    explain_indices: Dict[int, List[int]] = {}
    if os.path.exists(explain_indices_path):
        explain_indices = torch.load(explain_indices_path)
    else:
        print(f"  [warn] explain_indices not found: explain/score split unavailable")

    os.makedirs(requests_dir, exist_ok=True)

    scorers = ["llama"]
    if use_gpt4o:
        scorers.append("gpt4o")

    tasks = []
    for scorer in scorers:
        for task in ["detect", "fuzz"]:
            out_path = os.path.join(requests_dir, f"{combo_key}_{task}_{scorer}.jsonl")
            if os.path.exists(out_path):
                print(f"  [skip] Already exists: {out_path}")
                tasks.append((task, scorer, out_path, True))
            else:
                tasks.append((task, scorer, out_path, False))

    handles = {}
    for task, scorer, out_path, skip in tasks:
        if not skip:
            handles[(task, scorer)] = open(out_path, "w")

    written = {(t, s): 0 for t, s, _, skip in tasks if not skip}

    for feat_idx, explanation in explanations.items():
        feat_idx = int(feat_idx)
        explanation = clean_explanation(explanation)
        if feat_idx not in activations_dict:
            continue

        acts = activations_dict[feat_idx]
        ctxs = contexts_dict[feat_idx]

        used_indices = explain_indices.get(feat_idx, [])
        acts_s, ctxs_s, _ = sample_examples_stratified(
            acts, ctxs, n_score_examples, n_quantiles=10,
            exclude_indices=used_indices,
        )
        activating_texts = [
            decode_context(ctxs_s[i], tokenizer)
            for i in range(len(ctxs_s))
        ]
        ctx_half = ctxs_s.shape[1] // 2 if len(ctxs_s) > 0 else 16
        activating_fuzz_texts = [
            decode_context_with_highlight(ctxs_s[i], ctx_half, tokenizer, ctxs_s.shape[1])
            for i in range(len(ctxs_s))
        ]

        if len(ctxs_s) < n_score_examples:
            print(
                f"  [warn] feat {feat_idx}: insufficient score positives "
                f"({len(ctxs_s)}/{n_score_examples}) → skip"
            )
            continue

        if non_activating_contexts is not None:
            non_activating_texts = sample_non_activating_from_pool(
                non_activating_contexts, feat_idx, n_score_examples, tokenizer, seed=seed
            )
        else:
            non_activating_texts = sample_non_activating(
                contexts_dict, feat_idx, n_score_examples, tokenizer, seed=seed
            )

        if len(non_activating_texts) < n_score_examples:
            print(
                f"  [warn] feat {feat_idx}: insufficient negatives "
                f"({len(non_activating_texts)}/{n_score_examples}) → skip"
            )
            continue

        non_activating_texts = non_activating_texts[:n_score_examples]
        rng = random.Random(seed + feat_idx + 10000)
        non_activating_fuzz_texts = [
            add_random_highlight_to_text(text, rng)
            for text in non_activating_texts
        ]

        all_detect_texts = activating_texts + non_activating_texts
        # Paper-style reproduction target: 50 activating + 50 non-activating examples.
        # For fuzzing, non-activating contexts are converted into mislabeled marked-token
        # examples by marking one random non-empty token. The contexts themselves are
        # strict negatives when loaded from non_activating_contexts.pt.
        all_fuzz_texts = activating_fuzz_texts + non_activating_fuzz_texts

        for task, scorer, out_path, skip in tasks:
            if skip:
                continue

            model_id = NOVITA_MODEL if scorer == "llama" else OPENAI_MODEL

            if task == "detect":
                system = DETECTION_SYSTEM
                few_shot = DETECTION_FEW_SHOT
                task_examples = all_detect_texts
            else:
                system = FUZZING_SYSTEM
                few_shot = FUZZING_FEW_SHOT
                task_examples = all_fuzz_texts

            # Delphi-style scorer prompting is more stable with small batches.
            # Delphi few-shot examples use 5 examples -> 5 labels, so the default
            # chunk size is 5. parse_results reconstructs the original 100 labels.
            for chunk_idx, start in enumerate(range(0, len(task_examples), score_chunk_size)):
                chunk_examples = task_examples[start:start + score_chunk_size]
                n_total = len(chunk_examples)
                examples_str = build_scorer_examples_str(chunk_examples)

                user_content = (
                    f"Latent explanation: {explanation}\n\n"
                    f"Test examples:\n\n{examples_str}\n\n"
                    f"Return only one Python list.\n"
                    f"Do not explain. Do not apologize. Do not include any text before or after the list.\n"
                    f"The list must contain EXACTLY {n_total} integers, one for each example from Example 0 to Example {n_total - 1}.\n"
                    f"Each integer must be either 0 or 1. If unsure, guess 0 or 1.\n"
                )

                messages = [
                    {"role": "system", "content": system},
                    *few_shot,
                    {"role": "user", "content": user_content},
                ]

                custom_id = f"{combo_key}_feat{feat_idx}_{task}_{scorer}_chunk{chunk_idx}"
                record = {
                    "custom_id": custom_id,
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max(32, 8 * n_total),
                    "temperature": 0.0,
                }
                handles[(task, scorer)].write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                written[(task, scorer)] += 1

    for h in handles.values():
        h.close()

    out_paths = []
    for task, scorer, out_path, skip in tasks:
        if not skip:
            print(
                f"  Created {task}_{scorer} requests: {written.get((task, scorer), 0)} -> {out_path}"
                  )
            out_paths.append(out_path)

    return out_paths


# =========================================================
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",         type=str, choices=["explain", "score", "all"],
                        default="all", help="stage")
    parser.add_argument("--cache_dir",    type=str, default="./cache")
    parser.add_argument("--requests_dir", type=str, default="./requests")
    parser.add_argument("--scores_dir",   type=str, default="./scores",
                        help="Directory containing explanations.pt for the scoring")

    # model on/off
    parser.add_argument("--use_gpt2s",      action="store_true",   default=True)
    parser.add_argument("--no_gpt2s",       dest="use_gpt2s",      action="store_false")
    parser.add_argument("--use_pythia160m", action="store_true",   default=True)
    parser.add_argument("--no_pythia160m",  dest="use_pythia160m", action="store_false")
    parser.add_argument("--use_pythia410m", action="store_true",   default=True)
    parser.add_argument("--no_pythia410m",  dest="use_pythia410m", action="store_false")

    # scorer2 on/off
    parser.add_argument("--use_gpt4o",              action="store_true", default=False,
                        help="GPT-4o-mini for scorer2")
    parser.add_argument("--allow_approx_negatives", action="store_true", default=False,
                        help="approximate negatives")

    parser.add_argument("--n_features",         type=int, default=700)
    parser.add_argument("--n_explain_examples", type=int, default=40)
    parser.add_argument("--n_score_examples",   type=int, default=50)
    parser.add_argument("--score_chunk_size",   type=int, default=5,
                        help="Number of scorer examples per API request. Delphi few-shots use 5 examples.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_enabled = {
        "gpt2s": args.use_gpt2s,
        "pythia160m": args.use_pythia160m,
        "pythia410m": args.use_pythia410m,
    }

    for model_short, cfg in MODEL_CONFIGS.items():
        if not model_enabled.get(model_short, False):
            print(f"\n[skip] Model disabled: {model_short}")
            continue

        hf_name  = cfg["hf_name"]
        n_layers = cfg["n_layers"]

        print(f"\n{'='*60}")
        print(f"model: {model_short} ({hf_name})")
        print(f"{'='*60}")

        for layer_idx in range(n_layers):
            for mode in MODES:
                for k in TOPKS:
                    print(f"\n--- layer{layer_idx}/{mode}_k{k} ---")

                    if args.step in ("explain", "all"):
                        generate_explain_requests(
                            cache_dir=          args.cache_dir,
                            requests_dir=       args.requests_dir,
                            model_short=        model_short,
                            hf_name=            hf_name,
                            layer_idx=          layer_idx,
                            mode=               mode,
                            k=                  k,
                            n_features=         args.n_features,
                            n_explain_examples= args.n_explain_examples,
                            seed=               args.seed,
                        )

                    if args.step in ("score", "all"):
                        generate_score_requests(
                            cache_dir=              args.cache_dir,
                            requests_dir=           args.requests_dir,
                            scores_dir=             args.scores_dir,
                            model_short=            model_short,
                            hf_name=                hf_name,
                            layer_idx=              layer_idx,
                            mode=                   mode,
                            k=                      k,
                            n_score_examples=       args.n_score_examples,
                            use_gpt4o=              args.use_gpt4o,
                            seed=                   args.seed,
                            allow_approx_negatives= args.allow_approx_negatives,
                            score_chunk_size=       args.score_chunk_size,
                        )

    print("\nCompleted!")


if __name__ == "__main__":
    main()
