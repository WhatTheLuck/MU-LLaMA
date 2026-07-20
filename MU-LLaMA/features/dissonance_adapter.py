"""Adapter and cache contract for the repository's DissonanceSpectrum code."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch


FEATURE_VERSION = "mu-llama-ds-v1"
_MODULE = None


def _load_dissonance_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    source = Path(__file__).resolve().parents[2] / "DissonanceSpectrum" / "dissonance_spectrum.py"
    if not source.is_file():
        raise FileNotFoundError(f"DissonanceSpectrum implementation not found: {source}")
    spec = importlib.util.spec_from_file_location("mu_llama_dissonance_spectrum", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DissonanceSpectrum from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def _feature_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    feature = dict(config.get("feature", {}))
    feature.setdefault("reference_mode", "tonic+self")
    feature.setdefault("sr", 22050)
    feature.setdefault("fps", 0.0)
    feature.setdefault("hop_length", 1024)
    feature.setdefault("n_octaves", 8)
    feature.setdefault("bins_per_octave", 72)
    feature.setdefault("fmin", "C1")
    feature.setdefault("use_torch", True)
    feature.setdefault("return_details", True)
    # The underlying API gives fps precedence over hop_length. Experiments in
    # this repository vary hop_length, so zero fps is the intentional default.
    if "hop_length" in feature and "fps" not in config.get("feature", {}):
        feature["fps"] = 0.0
    feature["return_details"] = True
    return feature


def _config_hash(parameters: Dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _safe_id(audio_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", audio_id).strip("._")
    return (cleaned or "audio")[:96]


def load_tensor_file(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


class DissonanceFeatureAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.parameters = _feature_parameters(self.config)
        self.input_feature = self.config.get("input_feature", "dissonance_spectrum")
        if self.input_feature not in {"dissonance_spectrum", "processed_cqt"}:
            raise ValueError("input_feature must be dissonance_spectrum or processed_cqt")
        self.segment_seconds = self.config.get("segment_seconds", 60.0)
        self.segment_seconds = None if self.segment_seconds is None else float(self.segment_seconds)
        hash_parameters = {
            "feature": self.parameters,
            "segment_seconds": self.segment_seconds,
        }
        # Keep Stage 1 DS cache hashes stable; only the new CQT control needs
        # a distinct namespace because legacy payloads do not contain CQT.
        if self.input_feature == "processed_cqt":
            hash_parameters["input_feature"] = self.input_feature
        self.config_hash = _config_hash(hash_parameters)
        cache_root = Path(self.config.get("cache_root", "cache/dissonance"))
        self.cache_dir = cache_root / self.config_hash
        self.require_cache = bool(self.config.get("require_cache", True))

    @property
    def frequency_bins(self) -> int:
        return int(self.parameters["n_octaves"]) * int(self.parameters["bins_per_octave"])

    @property
    def tensor_key(self) -> str:
        return "processed_cqt" if self.input_feature == "processed_cqt" else "dissonance"

    def cache_path(self, audio_id: str, audio_path: str | Path) -> Path:
        resolved = Path(audio_path).resolve()
        identity = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
        label = resolved.stem or str(audio_id)
        return self.cache_dir / f"{_safe_id(label)}__{identity}.pt"

    def compute(self, audio_id: str, audio_path: str | Path) -> Dict[str, Any]:
        module = _load_dissonance_module()
        path = Path(audio_path).expanduser().resolve()
        calculation_path = path
        temporary_path = None
        try:
            if self.segment_seconds is not None and self.segment_seconds > 0:
                import soundfile as sf

                audio, original_sr = module.librosa.load(
                    path, sr=None, mono=True, duration=self.segment_seconds
                )
                temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temporary.close()
                temporary_path = Path(temporary.name)
                sf.write(temporary_path, audio, original_sr, subtype="FLOAT")
                calculation_path = temporary_path
            details = module.calculate_dissonance_spectrum(calculation_path, **self.parameters)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        spectrum = np.asarray(details["dissonance_spectrum"], dtype=np.float32)
        processed_cqt = np.asarray(details["processed_calc_cqt"], dtype=np.float32)
        if processed_cqt.shape != spectrum.shape:
            raise ValueError(
                f"Processed CQT shape {processed_cqt.shape} does not match DS shape {spectrum.shape}"
            )
        energy = module.calculate_dissonance_intensity(spectrum).astype(np.float32)
        configured_fps = float(self.parameters.get("fps", 0.0))
        configured_hop = int(self.parameters.get("hop_length", 1024))
        metadata = {
            "audio_id": str(audio_id),
            "audio_path": str(path),
            "sample_rate": int(details["sr"]),
            "hop_length": int(details["hop_length"]),
            "configured_fps": configured_fps,
            "effective_fps": float(details["fps"]),
            "configured_hop_length": configured_hop,
            "effective_hop_length": int(details["hop_length"]),
            "frequency_bins": int(spectrum.shape[0]),
            "time_frames": int(spectrum.shape[1]),
            "input_feature": self.input_feature,
            "bins_per_octave": int(self.parameters["bins_per_octave"]),
            "n_octaves": int(self.parameters["n_octaves"]),
            "feature_version": FEATURE_VERSION,
            "config_hash": self.config_hash,
            "backend": details.get("backend"),
            "segment_start_seconds": 0.0,
            "segment_duration_seconds": self.segment_seconds,
        }
        return {
            "dissonance": torch.from_numpy(spectrum),
            "processed_cqt": torch.from_numpy(processed_cqt),
            "energy": torch.from_numpy(energy),
            "metadata": metadata,
        }

    def validate(self, payload: Dict[str, Any]) -> None:
        spectrum = payload.get(self.tensor_key)
        metadata = payload.get("metadata", {})
        if not torch.is_tensor(spectrum) or spectrum.ndim != 2:
            raise ValueError(f"Cached {self.tensor_key} must be a 2-D tensor")
        if int(spectrum.shape[0]) != self.frequency_bins:
            raise ValueError(
                f"Cache has {spectrum.shape[0]} frequency bins; expected {self.frequency_bins}"
            )
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Dissonance cache feature_version mismatch")
        if metadata.get("config_hash") != self.config_hash:
            raise ValueError("Dissonance cache configuration hash mismatch")

    def load(self, audio_id: str, audio_path: str | Path) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DS is disabled; cache access is forbidden")
        path = self.cache_path(audio_id, audio_path)
        if path.is_file():
            payload = load_tensor_file(path)
            self.validate(payload)
            return payload
        if self.require_cache:
            raise FileNotFoundError(
                f"Missing Dissonance Spectrum cache for {audio_id}: {path}. "
                "Run tools/cache_dissonance.py first."
            )
        return self.compute(audio_id, audio_path)

    def save(self, payload: Dict[str, Any], audio_id: str, audio_path: str | Path) -> Path:
        self.validate(payload)
        target = self.cache_path(audio_id, audio_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        return target


def iter_audio_records(data_config: str | Path, audio_root: str | Path) -> Iterator[Tuple[str, Path, Dict[str, Any]]]:
    import yaml

    config_path = Path(data_config)
    with config_path.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle) or {}
    root = Path(audio_root)
    for meta_ref in dataset_config.get("META", []):
        meta_path = Path(meta_ref)
        if not meta_path.is_absolute() and not meta_path.is_file():
            meta_path = (config_path.parent / meta_path).resolve()
        with meta_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        for index, record in enumerate(records):
            audio_name = record.get("audio_path") or record.get("audio_name")
            if not audio_name:
                continue
            audio_path = Path(audio_name)
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_id = str(record.get("audio_id") or record.get("audio_name") or f"record_{index}")
            yield audio_id, audio_path.resolve(), record
