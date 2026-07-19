#!/usr/bin/env python3
"""Build resumable offline Dissonance Spectrum caches from experiment YAML."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dissonance_adapter import (  # noqa: E402
    DissonanceFeatureAdapter,
    iter_audio_records,
    load_tensor_file,
)
from util.config import load_config  # noqa: E402


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_config = config.get("model", {}).get("dissonance", {})
    adapter = DissonanceFeatureAdapter(ds_config)
    if not adapter.enabled:
        raise ValueError("Caching requires model.dissonance.enabled=true")
    data_config = config.get("data", {}).get("train_config")
    audio_root = config.get("data", {}).get("audio_root", "../MusicQA/audios")
    if not data_config:
        raise ValueError("data.train_config is required")

    failures_path = adapter.cache_dir / "failures.jsonl"
    manifest_path = adapter.cache_dir / "manifest.jsonl"
    processed = skipped = failed = visited = 0
    seen = set()
    for audio_id, audio_path, _ in iter_audio_records(data_config, audio_root):
        identity = str(audio_path)
        if identity in seen:
            continue
        seen.add(identity)
        if args.limit is not None and visited >= args.limit:
            break
        visited += 1
        target = adapter.cache_path(audio_id, audio_path)
        if target.is_file() and not args.force:
            try:
                adapter.validate(load_tensor_file(target))
                skipped += 1
                print(f"skip {audio_id}: {target}")
                continue
            except Exception as error:
                print(f"rebuild invalid cache {target}: {error}")
        try:
            payload = adapter.compute(audio_id, audio_path)
            saved = adapter.save(payload, audio_id, audio_path)
            processed += 1
            _append_jsonl(manifest_path, {
                "audio_id": audio_id,
                "audio_path": str(audio_path),
                "cache_path": str(saved),
                "shape": list(payload["dissonance"].shape),
                "metadata": payload["metadata"],
            })
            print(f"cached {audio_id}: {tuple(payload['dissonance'].shape)} -> {saved}")
        except Exception as error:
            failed += 1
            _append_jsonl(failures_path, {
                "audio_id": audio_id,
                "audio_path": str(audio_path),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            print(f"failed {audio_id}: {error}", file=sys.stderr)

    print(json.dumps({
        "cache_dir": str(adapter.cache_dir),
        "config_hash": adapter.config_hash,
        "cached": processed,
        "skipped": skipped,
        "failed": failed,
    }, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
