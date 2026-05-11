import os
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# =========================================================
# - Pythia: The Pile streaming
# - seq_len=2049
# - effective batch = micro_batch_size * grad_accum_steps = 64 sequences
# - loss = MSE only (aux_l2_coef=0)
# =========================================================
@dataclass
class TrainConfig:
    mode: str = "sae"  # "sae" or "transcoder"
    model_name: str = "EleutherAI/pythia-160m"  # gpt2, EleutherAI/pythia-410m
    layer_idx: int = 10
    device: str = "cuda:0"

    dataset_name: str = "EleutherAI/the_pile_deduplicated"
    dataset_split: str = "train"
    text_field: str = "text"
    seq_len: int = 2049     #
    token_budget: int = 8e10

    micro_batch_size: int = 1
    grad_accum_steps: int = 64
    num_workers: int = 0

    n_features: int = 49152
    k: int = 64

    lr: float = 5e-4
    weight_decay: float = 0.0
    max_steps: int = 61000
    log_every: int = 100
    save_every: int = 5_000
    save_dir: str = "./ckpts"

    recon_loss_coef: float = 1.0
    aux_l2_coef: float = 0.0

    seed: int = 42
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    model_family: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def normalize_rows_(W: torch.Tensor, eps: float = 1e-12) -> None:
    norms = W.norm(dim=1, keepdim=True).clamp_min(eps)
    W.div_(norms)


def topk_sparse_activation(z: torch.Tensor, k: int) -> torch.Tensor:
    vals, idx = torch.topk(z, k=k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, F.relu(vals))
    return out


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def infer_model_info(cfg: TrainConfig) -> TrainConfig:
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name)

    if hasattr(hf_cfg, "hidden_size"):
        cfg.d_model = int(hf_cfg.hidden_size)
    elif hasattr(hf_cfg, "n_embd"):
        cfg.d_model = int(hf_cfg.n_embd)
    else:
        raise ValueError(f"Unable to infer hidden size: {cfg.model_name}")

    if hasattr(hf_cfg, "num_hidden_layers"):
        cfg.n_layers = int(hf_cfg.num_hidden_layers)
    elif hasattr(hf_cfg, "n_layer"):
        cfg.n_layers = int(hf_cfg.n_layer)
    else:
        raise ValueError(f"Unable to infer number of layers: {cfg.model_name}")

    model_type = getattr(hf_cfg, "model_type", "").lower()
    arch = " ".join(getattr(hf_cfg, "architectures", [])).lower()

    if model_type == "gpt2" or "gpt2" in arch:
        cfg.model_family = "gpt2"
    elif model_type == "gpt_neox" or "gptneox" in arch or "gpt_neox" in arch:
        cfg.model_family = "gpt_neox"
    else:
        raise ValueError(f"Unsupported model family. model_type={model_type}, arch={arch}")

    if not (0 <= cfg.layer_idx < cfg.n_layers):
        raise ValueError(f"layer_idx={cfg.layer_idx} is out of range for model with {cfg.n_layers} layers")

    return cfg


class TokenStreamDataset(IterableDataset):
    def __init__(self, cfg: TrainConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __iter__(self):
        ds = load_dataset(self.cfg.dataset_name, split=self.cfg.dataset_split, streaming=True)
        buffer = []
        seen_tokens = 0

        for ex in ds:
            text = ex.get(self.cfg.text_field, None)
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
                buffer = buffer[self.cfg.seq_len :]
                seen_tokens += len(chunk)
                yield torch.tensor(chunk, dtype=torch.long)
                if seen_tokens >= self.cfg.token_budget:
                    return


class ActivationExtractor(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg   = cfg
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        self.model.eval().to(cfg.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._cache = {}
        self._hooks = []

        if cfg.model_family == "gpt2":
            self.block = self.model.transformer.h[cfg.layer_idx]
            self._register_gpt2_hooks()
        elif cfg.model_family == "gpt_neox":
            self.block = self.model.gpt_neox.layers[cfg.layer_idx]
            self._register_gpt_neox_hooks()
        else:
            raise ValueError(f"unsupported family: {cfg.model_family}")

    def _register_gpt2_hooks(self):
        def ln2_hook(module, inp, out):
            self._cache["mlp_in"] = out.detach()

        def mlp_hook(module, inp, out):
            self._cache["mlp_out"] = out.detach()

        self._hooks.append(self.block.ln_2.register_forward_hook(ln2_hook))
        self._hooks.append(self.block.mlp.register_forward_hook(mlp_hook))

    def _register_gpt_neox_hooks(self):
        def post_attn_ln_hook(module, inp, out):
            self._cache["mlp_in"] = out.detach()

        def mlp_hook(module, inp, out):
            self._cache["mlp_out"] = out.detach()

        self._hooks.append(self.block.post_attention_layernorm.register_forward_hook(post_attn_ln_hook))
        self._hooks.append(self.block.mlp.register_forward_hook(mlp_hook))

    @torch.no_grad()
    def get_batch(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cache = {}
        _ = self.model(input_ids=input_ids)
        if "mlp_in" not in self._cache or "mlp_out" not in self._cache:
            raise RuntimeError("Unable to capture hook activation.")
        return self._cache["mlp_in"], self._cache["mlp_out"]


class TopKDictionary(nn.Module):
    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_in, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.k     = k
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)
        with torch.no_grad():
            normalize_rows_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W_enc + self.b_enc
        return topk_sparse_activation(z, self.k)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec


class SAEModel(nn.Module):
    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.dictionary = TopKDictionary(d_in, n_features, k)

    def forward(self, x: torch.Tensor):
        h     = self.dictionary.encode(x)
        x_hat = self.dictionary.decode(h)
        return x_hat, h


class Transcoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_features: int, k: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_in, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        self.k     = k
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)
        with torch.no_grad():
            normalize_rows_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W_enc + self.b_enc
        return topk_sparse_activation(z, self.k)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        h     = self.encode(x)
        y_hat = self.decode(h)
        return y_hat, h


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor):
    mse = F.mse_loss(pred, target)
    rel = mse / (target.pow(2).mean() + 1e-8)
    return mse, rel


def aux_l2_loss(model: nn.Module) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if "W_" in name:
            loss = loss + (p.float() ** 2).mean()
    return loss


def flatten_bt(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.d_model is None:
        raise ValueError("cfg.d_model must be set before build_model")
    if cfg.mode == "sae":
        return SAEModel(cfg.d_model, cfg.n_features, cfg.k).to(cfg.device)
    if cfg.mode == "transcoder":
        return Transcoder(cfg.d_model, cfg.d_model, cfg.n_features, cfg.k).to(cfg.device)
    raise ValueError(f"unsupported mode: {cfg.mode}")


def normalize_decoder(model: nn.Module):
    with torch.no_grad():
        if hasattr(model, "dictionary"):
            normalize_rows_(model.dictionary.W_dec)
        else:
            normalize_rows_(model.W_dec)


def save_checkpoint(cfg: TrainConfig, model: nn.Module, step: int, loss: float, rel: float, name: Optional[str] = None):
    os.makedirs(cfg.save_dir, exist_ok=True)
    if name is None:
        name = f"{cfg.mode}_layer{cfg.layer_idx}_k{cfg.k}_step{step}_{loss:.4f}_{rel:.4f}.pt"
    path = os.path.join(cfg.save_dir, name)

    payload = {
        "cfg": {
            "mode":                 cfg.mode,
            "model_name":           cfg.model_name,
            "model_family":         cfg.model_family,
            "layer_idx":            cfg.layer_idx,
            "d_model":              cfg.d_model,
            "n_layers":             cfg.n_layers,
            "n_features":           cfg.n_features,
            "k":                    cfg.k,
            "dataset_name":         cfg.dataset_name,
            "seq_len":              cfg.seq_len,
            "effective_batch_size": cfg.micro_batch_size * cfg.grad_accum_steps,
            "aux_l2_coef":          cfg.aux_l2_coef,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    print(f"[saved] {path}")
    return path


def train_step(cfg: TrainConfig) -> Optional[str]:
    cfg = infer_model_info(cfg)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset   = TokenStreamDataset(cfg, tokenizer)
    loader    = DataLoader(dataset, batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers)

    extractor = ActivationExtractor(cfg)
    model     = build_model(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=5e-5)

    print(
        f"[start] mode={cfg.mode}, model={cfg.model_name}, layer={cfg.layer_idx}, k={cfg.k}, "
        f"seq_len={cfg.seq_len}, micro_batch={cfg.micro_batch_size}, "
        f"grad_accum={cfg.grad_accum_steps}, effective_batch={cfg.micro_batch_size * cfg.grad_accum_steps}, "
        f"dataset={cfg.dataset_name}, aux_l2={cfg.aux_l2_coef}"
    )

    optimizer.zero_grad(set_to_none=True)
    opt_step   = 0
    micro_step = 0
    last_path  = None
    accum_logs = []

    for batch_ids in loader:
        micro_step += 1
        batch_ids = batch_ids.to(cfg.device)

        with torch.no_grad():
            mlp_in, mlp_out = extractor.get_batch(batch_ids)

        if cfg.mode == "sae":
            x = flatten_bt(mlp_out.float())
            pred, h = model(x)
            target = x
        else:
            x = flatten_bt(mlp_in.float())
            y = flatten_bt(mlp_out.float())
            pred, h = model(x)
            target = y

        recon, recon_rel = reconstruction_loss(pred, target)
        reg = aux_l2_loss(model) if cfg.aux_l2_coef > 0 else torch.tensor(0.0, device=cfg.device)
        loss = cfg.recon_loss_coef * recon + cfg.aux_l2_coef * reg
        (loss / cfg.grad_accum_steps).backward()

        with torch.no_grad():
            active_per_token = (h > 0).sum(dim=-1).float().mean().item()
            accum_logs.append({
                "loss": loss.item(),
                "recon": recon.item(),
                "recon_rel": recon_rel.item(),
                "reg": reg.item(),
                "active_per_token": active_per_token,
            })

        del mlp_in, mlp_out, pred, h, target
        if cfg.mode == "sae":
            del x
        else:
            del x, y

        if micro_step % cfg.grad_accum_steps != 0:
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        normalize_decoder(model)
        opt_step += 1

        mean_loss = sum(d["loss"] for d in accum_logs) / len(accum_logs)
        mean_rel = sum(d["recon_rel"] for d in accum_logs) / len(accum_logs)
        mean_active = sum(d["active_per_token"] for d in accum_logs) / len(accum_logs)
        accum_logs = []

        if opt_step % cfg.log_every == 0:
            print(
                f"mode={cfg.mode} k={cfg.k} layer={cfg.layer_idx} step={opt_step} "
                f"loss={mean_loss:.6f} recon_rel={mean_rel:.6f} active={mean_active:.2f}"
            )

        if opt_step % cfg.save_every == 0:
            last_path = save_checkpoint(cfg, model, opt_step, mean_loss, mean_rel)

        if opt_step == cfg.max_steps:
            last_path = save_checkpoint(cfg, model, opt_step, mean_loss, mean_rel)

        if opt_step >= cfg.max_steps:
            break

    if last_path is None and 'mean_loss' in locals() and 'mean_rel' in locals():
        last_path = save_checkpoint(cfg, model, opt_step, mean_loss, mean_rel)

    print(f"[done] last_path={last_path}")
    return last_path


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",       type=str,   default="EleutherAI/pythia-160m")
    p.add_argument("--layers",           type=str,   default="10")
    p.add_argument("--topks",            type=str,   default="32")
    p.add_argument("--modes",            type=str,   default="sae")
    p.add_argument("--device",           type=str,   default="cuda:0")

    p.add_argument("--dataset_name",     type=str,   default="EleutherAI/pile")
    p.add_argument("--dataset_split",    type=str,   default="train")
    p.add_argument("--text_field",       type=str,   default="text")
    p.add_argument("--seq_len",          type=int,   default=2049)
    p.add_argument("--token_budget",     type=int,   default=8e9)
    p.add_argument("--micro_batch_size", type=int,   default=1)
    p.add_argument("--grad_accum_steps", type=int,   default=64)

    p.add_argument("--n_features",       type=int,   default=49152)
    p.add_argument("--lr",               type=float, default=5e-4)
    p.add_argument("--weight_decay",     type=float, default=0.0)
    p.add_argument("--max_steps",        type=int,   default=140000)
    p.add_argument("--log_every",        type=int,   default=100)
    p.add_argument("--save_every",       type=int,   default=5000)
    p.add_argument("--save_dir_root",    type=str,   default="./ckpts")
    p.add_argument("--aux_l2_coef",      type=float, default=0.0)
    p.add_argument("--seed",             type=int,   default=42)
    args = p.parse_args()

    layers = parse_int_list(args.layers)
    topks  = parse_int_list(args.topks)
    modes  = parse_str_list(args.modes)

    model_short = sanitize_model_name(args.model_name)

    for mode in modes:
        for k in topks:
            for layer in layers:
                save_dir = os.path.join(args.save_dir_root, f"{mode}_ckpt_{model_short}")
                cfg = TrainConfig(
                    mode=            mode,
                    model_name=      args.model_name,
                    layer_idx=       layer,
                    device=          args.device,
                    dataset_name=    args.dataset_name,
                    dataset_split=   args.dataset_split,
                    text_field=      args.text_field,
                    seq_len=         args.seq_len,
                    token_budget=    args.token_budget,
                    micro_batch_size=args.micro_batch_size,
                    grad_accum_steps=args.grad_accum_steps,
                    n_features=      args.n_features,
                    k=               k,
                    lr=              args.lr,
                    weight_decay=    args.weight_decay,
                    max_steps=       args.max_steps,
                    log_every=       args.log_every,
                    save_every=      args.save_every,
                    save_dir=        save_dir,
                    aux_l2_coef=     args.aux_l2_coef,
                    seed=            args.seed,
                )
                train_step(cfg)


if __name__ == "__main__":
    main()
