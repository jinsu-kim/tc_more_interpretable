# When and Why Transcoders Are More Interpretable

Transcoders (TCs) have been reported to outperform Sparse Autoencoders (SAEs) on automated interpretability benchmarks — but the structural reasons for this gap remain poorly understood. This repository contains code and experiments for our paper investigating when and why this advantage holds, using a layer-wise analysis across GPT2-small and Pythia-160M/410M.

## Key Findings
* TC's interpretability advantage is layer- and model-dependent, not a general property
* Decoder subspaces of SAE and TC are nearly identical — the gap is not explained by dictionary geometry
* TC features activate over more geometrically consistent inputs (higher sim<sub>x</sub>), and this consistency correlates significantly with per-feature auto-interp scores
* FFN neighborhood distortion (sim<sub>in</sub> − sim<sub>out</sub>) directionally aligns with both the interpretability gap and the consistency gap across layers

## Execution Order

### 1. Train sparse models

Train SAEs or Transcoders and save checkpoints.

```bash
python train_all.py \
  --model_name gpt2 \
  --save_every 3000 \
  --max_steps 12000 \
  --dataset_name EleutherAI/the_pile_deduplicated \
  --log_every 10 \
  --grad_accum_steps 32 \
  --layers 3 \
  --device cuda:1 \
  --modes transcoder \
  --topks 64 \
  --seq_len 1024
```

---

### 2. Save the best checkpoints

Save the selected checkpoints under `./ckpts_best`.

Example:

```text
./ckpts_best/sae_ckpt_pythia160m/sae_layer3_k32_best.pt
./ckpts_best/tc_ckpt_pythia160m/tc_layer3_k32_best.pt
```

---

### 3. Cache feature activations and contexts

Run the base LM and sparse models to collect:
- activating token indices
- activation values
- context windows
- strict non-activating contexts

```bash
python cache_activations_quantile_ready.py \
  --model pythia160m \
  --ckpt_dir ./ckpts_best \
  --save_dir ./cache \
  --dataset_name EleutherAI/the_pile_deduplicated \
  --seq_len 2049 \
  --ctx_len 32 \
  --token_budget 10000000 \
  --batch_size 1 \
  --max_examples_per_feature 2000 \
  --device cuda:1 \
  --mode sae
```

---

### 4. Generate explainer requests

Construct prompts for the explainer LLM.

```bash
python generate_requests.py \
  --step explain \
  --cache_dir ./cache \
  --requests_dir ./requests \
  --no_gpt2s \
  --no_pythia410m \
  --n_score_examples 50 \
  --n_features 700 \
  --n_explain_examples 40
```

---

### 5. Run the explainer model

Submit explainer requests to the API provider.

Explainer model:
- LLaMA 3.3 70B Instruct
- Hosted on Novita AI

```bash
python call_api_fixed.py \
  --step explain \
  --requests_dir requests \
  --results_dir results \
  --novita_key YOUR_API_KEY
```

---

### 6. Parse explainer outputs

Parse raw explainer responses and construct `explanations.pt`.

```bash
python parse_results.py \
  --results_dir results \
  --scores_dir scores \
  --n_score_examples 50 \
  --score_chunk_size 5 \
  --scorer llama \
  --final_n 500
```

---

### 7. Generate scorer requests

Construct detection and fuzzing evaluation requests.

The same LLaMA 3.3 70B model is used as the scorer.

```bash
python generate_requests.py \
  --step score \
  --cache_dir cache \
  --requests_dir requests \
  --scores_dir scores \
  --no_pythia160m \
  --no_pythia410m \
  --n_score_examples 50 \
  --score_chunk_size 5
```

---

### 8. Run the scorer model

Only unprocessed requests in `requests/` will be submitted.

```bash
python call_api_fixed.py \
  --step score \
  --requests_dir requests \
  --results_dir results \
  --novita_key YOUR_API_KEY
```

---

### 9. Compute detection and fuzzing scores

```bash
python parse_results_layerwise_repro_chunk5_fixed.py \
  --results_dir results \
  --scores_dir scores \
  --n_score_examples 50 \
  --score_chunk_size 5 \
  --scorer llama \
  --final_n 500 \
  --metric all
```