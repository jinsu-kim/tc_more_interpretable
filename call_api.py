import os
import json
import asyncio
import argparse
import time
import re
from typing import List, Optional, Set

import aiohttp
from openai import OpenAI

NOVITA_BASE_URL = "https://api.novita.ai/openai/v1"

NOVITA_CONCURRENCY = 15
NOVITA_RPM_LIMIT = 920
NOVITA_SLEEP = 60 / NOVITA_RPM_LIMIT

RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

ACTIVE_BATCH_STATUSES = {
    "validating",
    "in_progress",
    "finalizing",
    "submitted",
    "running",
    "queued",
}
TERMINAL_BATCH_STATUSES = {
    "completed",
    "failed",
    "expired",
    "cancelled",
    "cancelling",
}


# =========================================================
# =========================================================
def require_api_key(api_key: Optional[str], name: str) -> str:
    if not api_key:
        raise ValueError(f"{name} is not set")
    return api_key


def read_jsonl(path: str) -> List[dict]:
    records = []
    bad = 0
    with open(path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                continue
    if bad > 0:
        print(f"[warn] skipped {bad} bad jsonl lines: {path}")
    return records


def parse_int_filter(s: Optional[str]) -> Optional[Set[int]]:
    if s is None or str(s).strip() == "":
        return None
    return {int(x.strip()) for x in str(s).split(",") if x.strip()}


def parse_str_filter(s: Optional[str]) -> Optional[Set[str]]:
    if s is None or str(s).strip() == "":
        return None
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}


def file_matches_layer_topk_mode(
    fname: str,
    layers: Optional[Set[int]],
    topks: Optional[Set[int]],
    modes: Optional[Set[str]],
) -> bool:
    m = re.search(r"_layer(\d+)_(sae|tc|transcoder)_k(\d+)_", fname)
    if not m:
        return False

    layer = int(m.group(1))
    mode = m.group(2).lower()
    if mode == "transcoder":
        mode = "tc"
    k = int(m.group(3))

    if layers is not None and layer not in layers:
        return False
    if topks is not None and k not in topks:
        return False
    if modes is not None and mode not in modes:
        return False
    return True


def limit_records_to_first_n_features(records: List[dict], max_features: Optional[int]) -> List[dict]:
    if max_features is None or max_features <= 0:
        return records

    feat_pat = re.compile(r"_feat(\d+)_")
    keep_order = []
    keep_set = set()

    for r in records:
        cid = r.get("custom_id", "")
        m = feat_pat.search(cid)
        if not m:
            continue
        feat = int(m.group(1))
        if feat not in keep_set:
            keep_set.add(feat)
            keep_order.append(feat)
            if len(keep_order) >= max_features:
                break

    filtered = []
    for r in records:
        cid = r.get("custom_id", "")
        m = feat_pat.search(cid)
        if not m:
            continue
        if int(m.group(1)) in keep_set:
            filtered.append(r)

    print(
        f"[limit features] max_features={max_features} | "
        f"kept_features={len(keep_set)} | records={len(records)} -> {len(filtered)}"
    )
    return filtered


# =========================================================
# =========================================================
def sort_jsonl_by_custom_id(path: str):
    if not os.path.exists(path):
        return

    lines = read_jsonl(path)
    lines.sort(key=lambda x: x.get("custom_id", ""))

    with open(path, "w") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[sorted] {path}")


# =========================================================
# Novita async
# =========================================================
async def call_novita_single(session, record, api_key, semaphore):
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": record["model"],
            "messages": record["messages"],
            "max_tokens": record.get("max_tokens", 300),
            "temperature": record.get("temperature", 0.0),
        }

        for attempt in range(RETRY_ATTEMPTS):
            try:
                await asyncio.sleep(NOVITA_SLEEP)

                async with session.post(
                    f"{NOVITA_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"raw_text": await resp.text()}

                    if resp.status == 429:
                        wait = RETRY_DELAY * (attempt + 1)
                        print(f"[rate limit] wait {wait}s")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status != 200:
                        return {
                            "custom_id": record["custom_id"],
                            "error": str(data),
                            "response": None,
                        }

                    return {
                        "custom_id": record["custom_id"],
                        "error": None,
                        "response": data["choices"][0]["message"]["content"],
                    }

            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return {
                        "custom_id": record["custom_id"],
                        "error": str(e),
                        "response": None,
                    }

    return {"custom_id": record["custom_id"], "error": "max retries", "response": None}


async def call_novita_batch(records: List[dict], api_key: str, out_path: str):
    semaphore = asyncio.Semaphore(NOVITA_CONCURRENCY)

    done_ids = set()
    if os.path.exists(out_path):
        for obj in read_jsonl(out_path):
            if obj.get("error") is None and obj.get("response") is not None:
                done_ids.add(obj["custom_id"])

    pending = [r for r in records if r["custom_id"] not in done_ids]

    print(f"[Novita] total={len(records)} done={len(done_ids)} remain={len(pending)}")

    if not pending:
        sort_jsonl_by_custom_id(out_path)
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    async with aiohttp.ClientSession() as session:
        tasks = [call_novita_single(session, r, api_key, semaphore) for r in pending]

        with open(out_path, "a") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

                if (i + 1) % 100 == 0:
                    print(f"  progress {i+1}/{len(pending)}")

    sort_jsonl_by_custom_id(out_path)


def run_novita(
    jsonl_path: str,
    out_path: str,
    api_key: Optional[str],
    max_features: Optional[int] = None,
):
    api_key = require_api_key(api_key, "NOVITA_API_KEY")

    records = read_jsonl(jsonl_path)
    records = limit_records_to_first_n_features(records, max_features)
    asyncio.run(call_novita_batch(records, api_key, out_path))


# =========================================================
# OpenAI Batch
# =========================================================
def get_openai_batch_status(api_key: str, batch_id: str) -> str:
    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    return batch.status


def submit_openai_batch(jsonl_path: str, api_key: Optional[str], batch_meta_path: str, max_features: Optional[int] = None):
    api_key = require_api_key(api_key, "OPENAI_API_KEY")

    #  Prevent duplicate submissions
    if os.path.exists(batch_meta_path):
        with open(batch_meta_path, "r") as f:
            meta = json.load(f)

        batch_id = meta.get("batch_id")
        saved_status = meta.get("status")

        if batch_id:
            try:
                current_status = get_openai_batch_status(api_key, batch_id)
                meta["status"] = current_status
                with open(batch_meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                if current_status in ACTIVE_BATCH_STATUSES or current_status == "completed":
                    print(f"[skip batch exists] {batch_id} status={current_status} meta={batch_meta_path}")
                    return

                if current_status in TERMINAL_BATCH_STATUSES:
                    print(f"[warn] existing batch is terminal: {batch_id} status={current_status}")
                    print("       delete the meta file if you intentionally want to resubmit")
                    return

            except Exception as e:
                print(f"[warn] could not retrieve existing batch status: {e}")
                if saved_status in ACTIVE_BATCH_STATUSES or saved_status == "completed":
                    print(f"[skip batch exists] saved_status={saved_status} meta={batch_meta_path}")
                    return

    client = OpenAI(api_key=api_key)

    records = read_jsonl(jsonl_path)
    records = limit_records_to_first_n_features(records, max_features)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for record in records:
            tmp.write(json.dumps({
                "custom_id": record["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": record["model"],
                    "messages": record["messages"],
                    "max_tokens": record.get("max_tokens", 300),
                    "temperature": record.get("temperature", 0.0),
                }
            }, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
    finally:
        os.unlink(tmp_path)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    meta = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "jsonl_path": jsonl_path,
        "status": batch.status,
        "time": time.time(),
    }

    os.makedirs(os.path.dirname(batch_meta_path), exist_ok=True)
    with open(batch_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OpenAI] submitted {batch.id} status={batch.status}")


def normalize_openai_batch_output(raw_bytes: bytes, out_path: str):
    """
    Output schema:
        {"custom_id": str, "error": str|None, "response": str|None}
    """
    text = raw_bytes.decode("utf-8")
    normalized = []

    for line in text.splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj.get("custom_id")
        error = obj.get("error")
        response_text = None

        if error is not None:
            normalized.append({
                "custom_id": custom_id,
                "error": str(error),
                "response": None,
            })
            continue

        response = obj.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}

        if status_code != 200:
            normalized.append({
                "custom_id": custom_id,
                "error": json.dumps(body, ensure_ascii=False),
                "response": None,
            })
            continue

        try:
            response_text = body["choices"][0]["message"]["content"]
            normalized.append({
                "custom_id": custom_id,
                "error": None,
                "response": response_text,
            })
        except Exception as e:
            normalized.append({
                "custom_id": custom_id,
                "error": f"parse_error: {e}; body={json.dumps(body, ensure_ascii=False)}",
                "response": None,
            })

    normalized.sort(key=lambda x: x.get("custom_id", ""))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for obj in normalized:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[normalized] {out_path}")


def check_and_download_openai_batch(batch_meta_path: str, out_path: str, api_key: Optional[str]):
    api_key = require_api_key(api_key, "OPENAI_API_KEY")

    if not os.path.exists(batch_meta_path):
        print(f"[skip] No batch meta: {batch_meta_path}")
        return False

    with open(batch_meta_path, "r") as f:
        meta = json.load(f)

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(meta["batch_id"])

    print(f"[Batch] {batch.id} status={batch.status}")

    meta["status"] = batch.status
    meta["checked_time"] = time.time()

    if batch.status == "completed":
        if os.path.exists(out_path):
            print(f"[skip] already downloaded: {out_path}")
        else:
            content = client.files.content(batch.output_file_id)
            raw = content.read()
            normalize_openai_batch_output(raw, out_path)

        meta["status"] = "completed"
        meta["output_path"] = out_path

        with open(batch_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return True

    if getattr(batch, "error_file_id", None):
        meta["error_file_id"] = batch.error_file_id

    with open(batch_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return False


# =========================================================
# main
# =========================================================
def should_process_file(fname: str, step: str, use_gpt4o: bool) -> bool:
    if not fname.endswith(".jsonl"):
        return False

    if step == "explain":
        return "_explain.jsonl" in fname

    if step == "score":
        if "_detect_" not in fname and "_fuzz_" not in fname:
            return False
        if "gpt4o" in fname and not use_gpt4o:
            return False
        return True

    if step == "download_batch":
        return "gpt4o" in fname and ("_detect_" in fname or "_fuzz_" in fname)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["explain", "score", "download_batch"])
    parser.add_argument("--requests_dir", default="./requests")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--batch_meta_dir", default="./batch_meta")
    parser.add_argument("--novita_key", default=os.environ.get("NOVITA_API_KEY"))
    parser.add_argument("--openai_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--use_gpt4o", action="store_true")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to process, e.g. 0,5,10")
    parser.add_argument("--topks", type=str, default=None,
                        help="Comma-separated k values to process, e.g. 32,64")
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated modes to process: sae,tc")
    parser.add_argument("--max_features", type=int, default=None,
                        help="Keep only the first N features within each request file")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.batch_meta_dir, exist_ok=True)

    if not os.path.isdir(args.requests_dir):
        raise FileNotFoundError(f"requests_dir not found: {args.requests_dir}")

    files = sorted(os.listdir(args.requests_dir))
    target_layers = parse_int_filter(args.layers)
    target_topks = parse_int_filter(args.topks)
    target_modes = parse_str_filter(args.modes)

    if target_layers is not None:
        print(f"[filter] layers={sorted(target_layers)}")
    if target_topks is not None:
        print(f"[filter] topks={sorted(target_topks)}")
    if target_modes is not None:
        print(f"[filter] modes={sorted(target_modes)}")

    for fname in files:
        if not should_process_file(fname, args.step, args.use_gpt4o):
            continue

        if not file_matches_layer_topk_mode(fname, target_layers, target_topks, target_modes):
            continue

        in_path = os.path.join(args.requests_dir, fname)
        out_path = os.path.join(args.results_dir, fname.replace(".jsonl", "_out.jsonl"))

        if args.step == "explain":
            print(f"[Explain] {fname}")
            run_novita(in_path, out_path, args.novita_key, args.max_features)

        elif args.step == "score":
            if "gpt4o" in fname:
                meta_path = os.path.join(args.batch_meta_dir, fname.replace(".jsonl", "_batch.json"))
                print(f"[Batch submit] {fname}")
                submit_openai_batch(in_path, args.openai_key, meta_path, args.max_features)
            else:
                print(f"[Novita score] {fname}")
                run_novita(in_path, out_path, args.novita_key, args.max_features)

        elif args.step == "download_batch":
            meta_path = os.path.join(args.batch_meta_dir, fname.replace(".jsonl", "_batch.json"))
            print(f"[Batch check] {fname}")
            check_and_download_openai_batch(meta_path, out_path, args.openai_key)

    print("Completed")


if __name__ == "__main__":
    main()
