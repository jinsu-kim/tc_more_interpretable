import os
import gc
import math
import argparse
import heapq
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer



@dataclass
class CacheConfig:
    model_name:                         str = "gpt2"
    mode:                               str = "sae"
    device:                             str = "cuda:0"

    dataset_name:                       str = "EleutherAI/the_pile_deduplicated"
    dataset_split:                      str = "train"
    text_field:                         str = "text"
    seq_len:                            int = 128
    ctx_len:                            int = 32
    token_budget:                       int = 1e8
    batch_size:                         int = 32

    ckpt_dir:                           str = "~/PycharmProjects/identifiability/src/ckpts"
    save_dir:                           str = "./cache"

    min_examples:                       int = 200
    max_examples_per_feature:           int = 2000

    # strict negative context for scorer
    n_non_activating_per_feature:       int = 200
    non_activating_threshold:           float = 0.0
    negative_feature_chunk_size:        int = 512
    negative_max_per_feature_per_batch: int = 4

    # Use RAM to reduce Disk I/O
    use_ram_token_store:                bool = True
    n_features:                         int = 49_152

    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    model_family: Optional[str] = None



def safe_torch_load(path: str, map_location="cpu"):

    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"  [corrupt] torch.load failed: {path} ({type(e).__name__}: {e})")
        return None


def valid_torch_file(path: str) -> bool:

    if not os.path.exists(path):
        return False
    if os.path.getsize(path) <= 0:
        print(f"  corrupt file: {path}")
        return False
    return safe_torch_load(path, map_location="cpu") is not None


def valid_cache_file_set(save_dir: str, filenames: List[str]) -> bool:
    """
    Used to determine whether cache generation is complete.

    Return True only if all required files exist and can be successfully
    loaded with torch.load. If any file is corrupted, return False so that
    the corresponding layer/k cache is regenerated.
    """
    ok = True
    for name in filenames:
        path = os.path.join(save_dir, name)
        if not valid_torch_file(path):
            ok = False
    return ok


def remove_cache_file_set(save_dir: str, filenames: List[str]):

    for name in filenames:
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"  [remove] {path}")
            except Exception as e:
                print(f"  [warn] removal failed: {path} ({type(e).__name__}: {e})")

def infer_model_info(cfg: CacheConfig) -> CacheConfig:
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name)

    if hasattr(hf_cfg, "hidden_size"):
        cfg.d_model = int(hf_cfg.hidden_size)
    elif hasattr(hf_cfg, "n_embd"):
        cfg.d_model = int(hf_cfg.n_embd)
    else:
        raise ValueError(f"Unable to infer hidden size for model: {cfg.model_name}")

    if hasattr(hf_cfg, "num_hidden_layers"):
        cfg.n_layers = int(hf_cfg.num_hidden_layers)
    elif hasattr(hf_cfg, "n_layer"):
        cfg.n_layers = int(hf_cfg.n_layer)
    else:
        raise ValueError(f"Unable to infer number of layers for model: {cfg.model_name}")

    model_type = getattr(hf_cfg, "model_type", "").lower()
    arch = " ".join(getattr(hf_cfg, "architectures", [])).lower()

    if model_type == "gpt2" or "gpt2" in arch:
        cfg.model_family = "gpt2"
    elif model_type == "gpt_neox" or "gptneox" in arch or "gpt_neox" in arch:
        cfg.model_family = "gpt_neox"
    else:
        raise ValueError(f"Unsupported model family: {model_type}")

    return cfg


def get_short_name(model_name: str) -> str:
    mapping = {
        "gpt2": "gpt2s",
        "EleutherAI/pythia-160m": "pythia160m",
        "EleutherAI/pythia-410m": "pythia410m",
        "pythia-160m": "pythia160m",
        "pythia-410m": "pythia410m",
        "pythia160m": "pythia160m",
        "pythia410m": "pythia410m",
    }
    return mapping.get(model_name, model_name.replace("/", "_").replace("-", "_"))


def resolve_model_name(model_name: str) -> str:
    mapping = {
        "pythia-160m": "EleutherAI/pythia-160m",
        "pythia-410m": "EleutherAI/pythia-410m",
        "pythia160m": "EleutherAI/pythia-160m",
        "pythia410m": "EleutherAI/pythia-410m",
    }
    return mapping.get(model_name, model_name)


# =========================================================
# =========================================================
class TokenStreamDataset(IterableDataset):
    def __init__(self, cfg: CacheConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __iter__(self):
        ds = load_dataset(
            self.cfg.dataset_name,
            split=self.cfg.dataset_split,
            streaming=True,
        )
        buffer = []
        seen_tokens = 0

        for ex in ds:
            text = ex.get(self.cfg.text_field, "")
            if not text or not text.strip():
                continue

            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]

            if len(ids) == 0:
                continue

            buffer.extend(ids)

            while len(buffer) >= self.cfg.seq_len:
                chunk = buffer[: self.cfg.seq_len]
                buffer = buffer[self.cfg.seq_len:]
                seen_tokens += len(chunk)
                yield torch.tensor(chunk, dtype=torch.long)

                if seen_tokens >= self.cfg.token_budget:
                    return


def make_loader(cfg: CacheConfig, tokenizer) -> DataLoader:
    dataset = TokenStreamDataset(cfg, tokenizer)
    return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)


# =========================================================
# =========================================================
def ensure_token_memmap(cfg: CacheConfig, tokenizer, save_base_dir: str) -> Tuple[str, int]:

    short_name = get_short_name(cfg.model_name)
    memmap_dir = os.path.join(save_base_dir, short_name, "_token_store")
    os.makedirs(memmap_dir, exist_ok=True)

    mmap_path = os.path.join(memmap_dir, "tokens.uint32.dat")
    meta_path = os.path.join(memmap_dir, "tokens_meta.pt")

    if os.path.exists(mmap_path) and os.path.exists(meta_path):
        meta = safe_torch_load(meta_path, map_location="cpu")
        compatible = (
            meta is not None
            and os.path.exists(mmap_path)
            and os.path.getsize(mmap_path) >= int(cfg.token_budget) * np.dtype(np.uint32).itemsize
            and int(meta.get("token_budget", -1)) == int(cfg.token_budget)
            and int(meta.get("seq_len", -1)) == int(cfg.seq_len)
            and meta.get("dataset_name") == cfg.dataset_name
            and meta.get("dataset_split") == cfg.dataset_split
            and meta.get("text_field") == cfg.text_field
        )
        if compatible:
            print(f"Reusing existing token memmap: {mmap_path}")
            return mmap_path, int(meta["total_tokens"])

    total_tokens_target = cfg.token_budget
    arr = np.memmap(mmap_path, mode="w+", dtype=np.uint32, shape=(total_tokens_target,))

    write_pos = 0
    loader = make_loader(cfg, tokenizer)
    for batch in loader:
        flat = batch.reshape(-1).numpy().astype(np.uint32, copy=False)
        n = flat.shape[0]
        if write_pos + n > total_tokens_target:
            n = total_tokens_target - write_pos
            flat = flat[:n]
        arr[write_pos: write_pos + n] = flat
        write_pos += n
        if write_pos % 2e6 < max(1, n):
            print(f"Tokens written: {write_pos:,} ")
        if write_pos >= total_tokens_target:
            break

    arr.flush()
    del arr

    torch.save(
        {
            "total_tokens":  write_pos,
            "token_budget":  cfg.token_budget,
            "seq_len":       cfg.seq_len,
            "ctx_len":       cfg.ctx_len,
            "dataset_name":  cfg.dataset_name,
            "dataset_split": cfg.dataset_split,
            "text_field":    cfg.text_field,
            "dtype":         "uint32",
        },
        meta_path,
    )
    print(f"token memmap completed: {write_pos:,} tokens")
    return mmap_path, write_pos


# =========================================================
# All-layer Activation Extractor
# =========================================================
class AllLayerExtractor(nn.Module):
    def __init__(self, cfg: CacheConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        self.model.eval().to(cfg.device)
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._hooks = []
        self._register_all_hooks()

    def _register_all_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._cache = {}

        if self.cfg.model_family == "gpt2":
            for layer_idx, block in enumerate(self.model.transformer.h):
                def make_in_hook(idx):
                    def hook(m, inp, out):
                        self._cache.setdefault(idx, {})["mlp_in"] = out.detach()
                    return hook

                def make_out_hook(idx):
                    def hook(m, inp, out):
                        self._cache.setdefault(idx, {})["mlp_out"] = out.detach()
                    return hook

                self._hooks.append(block.ln_2.register_forward_hook(make_in_hook(layer_idx)))
                self._hooks.append(block.mlp.register_forward_hook(make_out_hook(layer_idx)))

        elif self.cfg.model_family == "gpt_neox":
            for layer_idx, block in enumerate(self.model.gpt_neox.layers):
                def make_in_hook(idx):
                    def hook(m, inp, out):
                        self._cache.setdefault(idx, {})["mlp_in"] = out.detach()
                    return hook

                def make_out_hook(idx):
                    def hook(m, inp, out):
                        self._cache.setdefault(idx, {})["mlp_out"] = out.detach()
                    return hook

                self._hooks.append(block.post_attention_layernorm.register_forward_hook(make_in_hook(layer_idx)))
                self._hooks.append(block.mlp.register_forward_hook(make_out_hook(layer_idx)))

    @torch.no_grad()
    def get_all_layers(self, input_ids: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        self._cache = {}
        self.model(input_ids=input_ids)
        return self._cache


# =========================================================
# =========================================================
def topk_sparse_activation(z: torch.Tensor, k: int) -> torch.Tensor:
    vals, idx = torch.topk(z, k=k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, F.relu(vals))
    return out


class TopKDictionary(nn.Module):
    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.k     = k
        self.W_enc = nn.Parameter(torch.empty(d_in, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_sparse_activation(x @ self.W_enc + self.b_enc, self.k)


class SAEModel(nn.Module):
    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.dictionary = TopKDictionary(d_in, n_features, k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary.encode(x)


class Transcoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_features: int, k: int):
        super().__init__()
        self.k     = k
        self.W_enc = nn.Parameter(torch.empty(d_in, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_sparse_activation(x @ self.W_enc + self.b_enc, self.k)



def load_sparse_model(ckpt_path: str, d_model: int, device: str) -> Tuple[nn.Module, str, int]:
    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload["cfg"]
    mode = cfg_dict["mode"]
    k = cfg_dict["k"]
    n_features = cfg_dict["n_features"]

    if mode == "sae":
        model = SAEModel(d_model, n_features, k)
    elif mode in ("transcoder", "tc"):
        model = Transcoder(d_model, d_model, n_features, k)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, mode, k


# =========================================================
# =========================================================

def open_token_store(token_memmap_path: str, total_tokens: int, use_ram_token_store: bool):
    token_arr = np.memmap(token_memmap_path, mode="r", dtype=np.uint32, shape=(total_tokens,))
    if use_ram_token_store:
        print("    token store RAM preload ")
        token_arr = np.asarray(token_arr).copy()
        print(f"    token store RAM preloaded: {token_arr.nbytes / (1024 ** 2):.1f} MB")
    return token_arr

def extract_context_from_memmap(token_arr, global_idx: int, ctx_len: int, total_len: int) -> torch.Tensor:

    ctx_half  = ctx_len // 2
    src_start = max(0, global_idx - ctx_half)
    src_end   = min(total_len, global_idx + ctx_half)

    raw       = np.asarray(token_arr[src_start:src_end], dtype=np.int32)
    pad_left  = max(0, ctx_half - global_idx)
    pad_right = ctx_len - pad_left - raw.shape[0]

    if pad_left > 0 or pad_right > 0:
        raw = np.pad(raw, (pad_left, max(0, pad_right)), mode="constant", constant_values=0)

    raw       = raw[:ctx_len]
    return torch.from_numpy(raw.astype(np.int32, copy=False))


# =========================================================
# =========================================================
def update_heaps_from_hidden(
    cfg: CacheConfig,
    h_cpu: torch.Tensor,          # (B*T, n_features), CPU
    global_token_idx: int,
    feature_heaps: Dict[int, list],
    feature_freq: torch.Tensor,
    tie_counter_ref: List[int],
):
    max_ex = cfg.max_examples_per_feature
    BT = h_cpu.shape[0]

    for local_idx in range(BT):
        global_idx = global_token_idx + local_idx

        nonzero_mask = h_cpu[local_idx] > 0
        if not nonzero_mask.any():
            continue

        nonzero_features = nonzero_mask.nonzero(as_tuple=True)[0]
        nonzero_values = h_cpu[local_idx][nonzero_features]

        for feat_idx, val in zip(nonzero_features.tolist(), nonzero_values.tolist()):
            feature_freq[feat_idx] += 1
            heap = feature_heaps.setdefault(feat_idx, [])
            tie_counter_ref[0] += 1
            tc = tie_counter_ref[0]
            item = (float(val), tc, global_idx)

            if len(heap) < max_ex:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)


# =========================================================
# =========================================================
def save_cache_from_indices(
    cfg: CacheConfig,
    feature_heaps: Dict[int, list],
    feature_freq: torch.Tensor,
    save_dir: str,
    token_memmap_path: str,
    total_tokens: int,
) -> List[int]:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(feature_freq, os.path.join(save_dir, "freq.pt"))

    alive_features = (feature_freq >= cfg.min_examples).nonzero(as_tuple=True)[0].tolist()
    print(f"    alive features: {len(alive_features):,} / {cfg.n_features:,}")

    activations_dict: Dict[int, torch.Tensor] = {}
    indices_dict: Dict[int, torch.Tensor] = {}

    for feat_idx in alive_features:
        heap = feature_heaps.get(feat_idx, [])
        if not heap:
            continue
        vals = torch.tensor([item[0] for item in heap], dtype=torch.float32)
        idxs = torch.tensor([item[2] for item in heap], dtype=torch.long)
        order = torch.argsort(vals, descending=True)
        activations_dict[feat_idx] = vals[order]
        indices_dict[feat_idx] = idxs[order]

    torch.save(activations_dict, os.path.join(save_dir, "activations.pt"))
    torch.save(indices_dict,     os.path.join(save_dir, "indices.pt"))
    torch.save(alive_features,   os.path.join(save_dir, "alive_features.pt"))
    torch.save(
        {
            "max_examples_per_feature": cfg.max_examples_per_feature,
            "min_examples":             cfg.min_examples,
            "ctx_len":                  cfg.ctx_len,
            "token_budget":             cfg.token_budget,
            "sampling_note":            "Top active candidates sorted by activation. Use generate_requests.py for 10-quantile sampling.",
        },
        os.path.join(save_dir, "cache_meta.pt"),
    )


    token_arr = open_token_store(token_memmap_path, total_tokens, cfg.use_ram_token_store)
    contexts_dict: Dict[int, torch.Tensor] = {}
    for feat_idx in alive_features:
        idxs = indices_dict.get(feat_idx)
        if idxs is None or idxs.numel() == 0:
            continue
        ctxs = [extract_context_from_memmap(token_arr, int(idx.item()), cfg.ctx_len, total_tokens) for idx in idxs]
        contexts_dict[feat_idx] = torch.stack(ctxs, dim=0)

    torch.save(contexts_dict, os.path.join(save_dir, "token_contexts.pt"))
    print(f" Saved: {save_dir}")

    del token_arr
    del contexts_dict
    del indices_dict
    del activations_dict
    gc.collect()
    return alive_features



# =========================================================
# Strict negative context 
# =========================================================
def _negative_cache_complete(save_dir: str, target_n: int) -> bool:
    path = os.path.join(save_dir, "non_activating_contexts.pt")
    if not os.path.exists(path):
        return False
    obj = safe_torch_load(path, map_location="cpu")
    if obj is None:
        return False
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    return all(hasattr(v, "shape") and int(v.shape[0]) >= target_n for v in obj.values())


def save_non_activating_contexts_from_indices(
    cfg: CacheConfig,
    neg_indices: Dict[int, List[int]],
    save_dir: str,
    token_memmap_path: str,
    total_tokens: int,
):
    """
    Extract non-activating global_idx
    {feat_idx: Tensor[n_non_activating, ctx_len]}
    """
    os.makedirs(save_dir, exist_ok=True)
    token_arr = open_token_store(token_memmap_path, total_tokens, cfg.use_ram_token_store)

    contexts_dict: Dict[int, torch.Tensor] = {}
    for feat_idx, idxs in neg_indices.items():
        if not idxs:
            continue
        idxs = idxs[: cfg.n_non_activating_per_feature]
        ctxs = [
            extract_context_from_memmap(token_arr, int(global_idx), cfg.ctx_len, total_tokens)
            for global_idx in idxs
        ]
        if ctxs:
            contexts_dict[int(feat_idx)] = torch.stack(ctxs, dim=0)

    torch.save(contexts_dict, os.path.join(save_dir, "non_activating_contexts.pt"))
    torch.save(
        {
            "n_non_activating_per_feature": cfg.n_non_activating_per_feature,
            "non_activating_threshold":     cfg.non_activating_threshold,
            "ctx_len":                      cfg.ctx_len,
            "token_budget":                 cfg.token_budget,
            "sampling_note":                "Strict negatives: contexts verified with h[:, feat_idx] <= threshold during a separate negative pass.",
        },
        os.path.join(save_dir, "non_activating_meta.pt"),
    )

    del token_arr
    del contexts_dict
    gc.collect()


def collect_non_activating_for_combo(
    cfg: CacheConfig,
    mode: str,
    extractor: AllLayerExtractor,
    tokenizer,
    layer_records: List[Tuple[int, nn.Module, str, List[int]]],
    token_memmap_path: str,
    total_tokens: int,
):
    """
    Construct strict negative contexts for the scorer.

    For each feature, collect token positions where the sparse activation h[:, feat_idx] is below the threshold.
    After gathering the corresponding global indices, reconstruct contexts from the token memmap.
    """
    target_n = int(cfg.n_non_activating_per_feature)
    if target_n <= 0:
        print("  [skip] n_non_activating_per_feature <= 0")
        return

    active_records = []
    for layer_idx, sparse_model, save_dir, alive_features in layer_records:
        if not alive_features:
            continue
        if _negative_cache_complete(save_dir, target_n):
            print(f"  [skip] non_activating_contexts already completed: layer{layer_idx}")
            continue
        alive_features = [int(f) for f in alive_features]
        neg_indices = {int(f): [] for f in alive_features}
        active_records.append((layer_idx, sparse_model, save_dir, alive_features, neg_indices))

    if not active_records:
        print("No layers require negative processing")
        return

    print("\nCollecting strict non-activating contexts...")
    loader = make_loader(cfg, tokenizer)
    global_token_idx = 0
    threshold = float(cfg.non_activating_threshold)
    chunk_size = int(cfg.negative_feature_chunk_size)
    max_per_batch = int(cfg.negative_max_per_feature_per_batch)

    for batch_idx, batch in enumerate(loader):
        if all(
            all(len(neg_indices[f]) >= target_n for f in alive_features)
            for _, _, _, alive_features, neg_indices in active_records
        ):
            break

        batch = batch.to(cfg.device)
        B, T = batch.shape
        BT = B * T
        layer_cache = extractor.get_all_layers(batch)

        for layer_idx, sparse_model, _, alive_features, neg_indices in active_records:
            if all(len(neg_indices[f]) >= target_n for f in alive_features):
                continue
            if layer_idx not in layer_cache:
                continue

            if mode == "sae":
                x = layer_cache[layer_idx]["mlp_out"].reshape(-1, cfg.d_model).float()
            else:
                x = layer_cache[layer_idx]["mlp_in"].reshape(-1, cfg.d_model).float()

            with torch.no_grad():
                h = sparse_model.encode(x)
            del x

            need_feats = [f for f in alive_features if len(neg_indices[f]) < target_n]
            for start in range(0, len(need_feats), chunk_size):
                feat_chunk = need_feats[start: start + chunk_size]
                if not feat_chunk:
                    continue

                feat_tensor = torch.tensor(feat_chunk, dtype=torch.long, device=h.device)
                h_sub = h.index_select(1, feat_tensor)
                inactive = h_sub <= threshold

                for col, feat_idx in enumerate(feat_chunk):
                    rem = target_n - len(neg_indices[feat_idx])
                    if rem <= 0:
                        continue
                    cand = inactive[:, col].nonzero(as_tuple=True)[0]
                    if cand.numel() == 0:
                        continue
                    take = min(rem, max_per_batch, int(cand.numel()))

                    perm = torch.randperm(cand.numel(), device=cand.device)[:take]
                    chosen = cand[perm].detach().cpu().tolist()
                    neg_indices[feat_idx].extend([global_token_idx + int(i) for i in chosen])

                del feat_tensor, h_sub, inactive

            del h

        global_token_idx += BT
        if batch_idx % 50 == 0:
            status = []
            for layer_idx, _, _, alive_features, neg_indices in active_records:
                done = sum(len(neg_indices[f]) >= target_n for f in alive_features)
                status.append(f"layer{layer_idx}: {done}/{len(alive_features)}")
            print(f"  negative pass: {global_token_idx:,} / {total_tokens:,} tokens | " + " | ".join(status))

        del layer_cache, batch
        gc.collect()
        torch.cuda.empty_cache()

    for layer_idx, _, save_dir, alive_features, neg_indices in active_records:
        complete = sum(len(neg_indices[f]) >= target_n for f in alive_features)
        print(f"  layer{layer_idx}: complete {complete}/{len(alive_features)} features")
        save_non_activating_contexts_from_indices(
            cfg=cfg,
            neg_indices=neg_indices,
            save_dir=save_dir,
            token_memmap_path=token_memmap_path,
            total_tokens=total_tokens,
        )

    gc.collect()
    torch.cuda.empty_cache()

# =========================================================
# =========================================================
def run_caching_one_combo(
    cfg: CacheConfig,
    mode: str,
    k: int,
    extractor: AllLayerExtractor,
    tokenizer,
    ckpt_base_dir: str,
    save_base_dir: str,
    token_memmap_path: str,
    total_tokens: int,
    layers: Optional[List[int]] = None,
):
    short_name = get_short_name(cfg.model_name)
    target_layers = layers if layers is not None else list(range(cfg.n_layers))

    print(f"\n{'='*60}")
    print(f"combo: mode={mode}, k={k}, model={short_name}")
    print(f"{'='*60}")

    layer_info = []
    for layer_idx in target_layers:
        ckpt_folder = f"{mode}_ckpt_{short_name}"
        ckpt_file = f"{mode}_layer{layer_idx}_k{k}_best.pt"
        ckpt_path = os.path.join(os.path.expanduser(ckpt_base_dir), ckpt_folder, ckpt_file)
        save_dir = os.path.join(save_base_dir, short_name, f"layer{layer_idx}", f"{mode}_k{k}")

        positive_files = [
            "freq.pt",
            "activations.pt",
            "indices.pt",
            "alive_features.pt",
            "cache_meta.pt",
            "token_contexts.pt",
        ]
        negative_files = ["non_activating_contexts.pt", "non_activating_meta.pt"]

        positive_done = valid_cache_file_set(save_dir, positive_files)
        negative_done = (
            cfg.n_non_activating_per_feature <= 0
            or (
                valid_cache_file_set(save_dir, negative_files)
                and _negative_cache_complete(save_dir, cfg.n_non_activating_per_feature)
            )
        )

        if positive_done and negative_done:
            print(f"  [skip] Cache verified: layer{layer_idx}")
            continue

        if not positive_done:
            print(f"  [redo] positive cache is incomplete: layer{layer_idx}")
            remove_cache_file_set(save_dir, positive_files + negative_files)
        elif positive_done and not negative_done:
            print(f"  [redo] Positive cache is valid, but negative cache is incomplete or corrupted: layer{layer_idx}")
            remove_cache_file_set(save_dir, negative_files)
        if not os.path.exists(ckpt_path):
            print(f"  [skip] No Checkpoint: {ckpt_path}")
            continue

        sparse_model, _, _ = load_sparse_model(ckpt_path, cfg.d_model, device=cfg.device)
        layer_info.append((layer_idx, sparse_model, save_dir))
        print(f"  GPU Load: layer{layer_idx}/{mode}_k{k}")

    if not layer_info:
        print("  No layer to process")
        return

    layer_heaps: Dict[int, Dict[int, list]] = {layer_idx: {} for layer_idx, _, _ in layer_info}
    layer_freqs: Dict[int, torch.Tensor] = {layer_idx: torch.zeros(cfg.n_features, dtype=torch.long) for layer_idx, _, _ in layer_info}
    layer_ties: Dict[int, List[int]] = {layer_idx: [0] for layer_idx, _, _ in layer_info}

    loader = make_loader(cfg, tokenizer)
    global_token_idx = 0

    for batch_idx, batch in enumerate(loader):
        batch =       batch.to(cfg.device)
        B, T =        batch.shape
        layer_cache = extractor.get_all_layers(batch)

        for layer_idx, sparse_model, _ in layer_info:
            if layer_idx not in layer_cache:
                continue
            if mode == "sae":
                x = layer_cache[layer_idx]["mlp_out"].reshape(-1, cfg.d_model).float()
            else:
                x = layer_cache[layer_idx]["mlp_in"].reshape(-1, cfg.d_model).float()

            with torch.no_grad():
                h = sparse_model.encode(x)
            h_cpu = h.cpu()
            del h, x

            update_heaps_from_hidden(
                cfg=              cfg,
                h_cpu=            h_cpu,
                global_token_idx= global_token_idx,
                feature_heaps=    layer_heaps[layer_idx],
                feature_freq=     layer_freqs[layer_idx],
                tie_counter_ref=  layer_ties[layer_idx],
            )
            del h_cpu

        global_token_idx += B * T
        if batch_idx % 200 == 0:
            print(f"  Done: {global_token_idx:,} / {total_tokens:,} tokens")

        del layer_cache, batch
        gc.collect()
        torch.cuda.empty_cache()

    negative_layer_records = []
    for layer_idx, sparse_model, save_dir in layer_info:
        print(f"  layer{layer_idx}/{mode}_k{k}")
        alive_features = save_cache_from_indices(
            cfg=               cfg,
            feature_heaps=     layer_heaps[layer_idx],
            feature_freq=      layer_freqs[layer_idx],
            save_dir=          save_dir,
            token_memmap_path= token_memmap_path,
            total_tokens=      total_tokens,
        )
        negative_layer_records.append((layer_idx, sparse_model, save_dir, alive_features))
        del layer_heaps[layer_idx]
        del layer_freqs[layer_idx]
        gc.collect()
        torch.cuda.empty_cache()

    collect_non_activating_for_combo(
        cfg=cfg,
        mode=mode,
        extractor=extractor,
        tokenizer=tokenizer,
        layer_records=negative_layer_records,
        token_memmap_path=token_memmap_path,
        total_tokens=total_tokens,
    )

    for layer_idx, sparse_model, save_dir, _ in negative_layer_records:
        del sparse_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"combo done: mode={mode}, k={k}")


# =========================================================
# =========================================================
def run_caching(
    model_name:                         str,
    mode:                               str,
    ckpt_base_dir:                      str,
    save_base_dir:                      str,
    device:                             str = "cuda:0",
    token_budget:                       int = 1e7,
    batch_size:                         int = 32,
    min_examples:                       int = 200,
    max_examples_per_feature:           int = 2000,
    n_non_activating_per_feature:       int = 200,
    non_activating_threshold:           float = 0.0,
    negative_feature_chunk_size:        int = 512,
    negative_max_per_feature_per_batch: int = 4,
    dataset_name:                       str = "EleutherAI/the_pile_deduplicated",
    dataset_split:                      str = "train",
    text_field:                         str = "text",
    seq_len:                            int = 128,
    ctx_len:                            int = 32,
    layer_start:                        int = 0,
    layer_end: Optional[int] = None,
    use_ram_token_store: bool = False,
):
    cfg = CacheConfig(
        model_name=                         model_name,
        mode=                               mode,
        device=                             device,
        dataset_name=                       dataset_name,
        dataset_split=                      dataset_split,
        text_field=                         text_field,
        seq_len=                            seq_len,
        ctx_len=                            ctx_len,
        token_budget=                       token_budget,
        batch_size=                         batch_size,
        min_examples=                       min_examples,
        max_examples_per_feature=           max_examples_per_feature,
        n_non_activating_per_feature=       n_non_activating_per_feature,
        non_activating_threshold=           non_activating_threshold,
        negative_feature_chunk_size=        negative_feature_chunk_size,
        negative_max_per_feature_per_batch= negative_max_per_feature_per_batch,
        ckpt_dir=                           ckpt_base_dir,
        save_dir=                           save_base_dir,
        use_ram_token_store=                use_ram_token_store,
    )
    cfg = infer_model_info(cfg)

    layer_end_actual = layer_end if layer_end is not None else cfg.n_layers
    layer_end_actual = min(layer_end_actual, cfg.n_layers)
    layers_to_run = list(range(layer_start, layer_end_actual))

    short_name = get_short_name(model_name)
    print(f"\n{'='*60}")
    print(f"model: {model_name} ({short_name})")
    print(f"n_layers: {cfg.n_layers}, d_model: {cfg.d_model}")
    print(f"layers to process: {layer_start} ~ {layer_end_actual - 1}")
    print(f"use_ram_token_store: {cfg.use_ram_token_store}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_memmap_path, total_tokens = ensure_token_memmap(cfg, tokenizer, save_base_dir)

    extractor = AllLayerExtractor(cfg)
    for mode in [cfg.mode]:
        for k in [32, 64]:
            run_caching_one_combo(
                cfg=               cfg,
                mode=              mode,
                k=                 k,
                extractor=         extractor,
                tokenizer=         tokenizer,
                ckpt_base_dir=     ckpt_base_dir,
                save_base_dir=     save_base_dir,
                token_memmap_path= token_memmap_path,
                total_tokens=      total_tokens,
                layers=            layers_to_run,
            )

    print(f"\nCaching completed for {model_name} Layer {layer_start}~{layer_end_actual - 1}")


# =========================================================
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",                              type=str, default="gpt2")
    parser.add_argument("--mode",                               type=str, default="sae")
    parser.add_argument("--ckpt_dir",                           type=str, default="~/PycharmProjects/tc_more_interpretable/src/ckpts")
    parser.add_argument("--save_dir",                           type=str, default="./cache")
    parser.add_argument("--device",                             type=str, default="cuda:0")
    parser.add_argument("--dataset_name",                       type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--dataset_split",                      type=str, default="train")
    parser.add_argument("--text_field",                         type=str, default="text")
    parser.add_argument("--seq_len",                            type=int, default=128)
    parser.add_argument("--ctx_len",                            type=int, default=32)
    parser.add_argument("--token_budget",                       type=int, default=1e7)
    parser.add_argument("--batch_size",                         type=int, default=32)
    parser.add_argument("--min_examples",                       type=int, default=200)
    parser.add_argument("--max_examples_per_feature",           type=int, default=2000)
    parser.add_argument("--negative_feature_chunk_size",        type=int, default=512)
    parser.add_argument("--negative_max_per_feature_per_batch", type=int, default=4)
    parser.add_argument("--layer_start",                        type=int, default=0)
    parser.add_argument("--layer_end",                          type=int, default=None)

    parser.add_argument("--n_non_activating_per_feature",       type=int, default=200,
                        help="Number of strict non-activating contexts per feature")
    parser.add_argument("--non_activating_threshold",           type=float, default=0.0,
                        help="Treat h[:, feat_idx] <= threshold as non-activating")
    parser.add_argument("--use_ram_token_store", action="store_true",
                        help="Use RAM to reduce Disk I/O")
    args = parser.parse_args()

    model_name = resolve_model_name(args.model)
    run_caching(
        model_name=                         model_name,
        mode=                               args.mode,
        ckpt_base_dir=                      args.ckpt_dir,
        save_base_dir=                      args.save_dir,
        device=                             args.device,
        dataset_name=                       args.dataset_name,
        dataset_split=                      args.dataset_split,
        text_field=                         args.text_field,
        seq_len=                            args.seq_len,
        ctx_len=                            args.ctx_len,
        token_budget=                       args.token_budget,
        batch_size=                         args.batch_size,
        min_examples=                       args.min_examples,
        max_examples_per_feature=           args.max_examples_per_feature,
        n_non_activating_per_feature=       args.n_non_activating_per_feature,
        non_activating_threshold=           args.non_activating_threshold,
        negative_feature_chunk_size=        args.negative_feature_chunk_size,
        negative_max_per_feature_per_batch= args.negative_max_per_feature_per_batch,
        layer_start=                        args.layer_start,
        layer_end=                          args.layer_end,
        use_ram_token_store=                args.use_ram_token_store,
    )


if __name__ == "__main__":
    main()
