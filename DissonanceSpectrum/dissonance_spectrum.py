"""Standalone, accelerated Visual Dissonance spectrum API.

The public API is intentionally small::

    spectrum = calculate_dissonance_spectrum("audio.wav")
    intensity = calculate_dissonance_intensity(spectrum)

All project-specific helpers live in this file.  NumPy and librosa are required;
PyTorch is optional and is used for the matrix convolution when requested.
"""

from __future__ import annotations

import json
import warnings
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
REF_AUDIO_DIR = MODULE_DIR / "ref_audio"
CURVE_DIR = MODULE_DIR / "dissonance_curves"
TONIC_OPTIONS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TONALITY_OPTIONS = ("major", "minor")
REFERENCE_MODES = ("specified", "tonic", "tonic_self", "tonic+self", "self", "context_window")
REFERENCE_MODE_ALIASES = {"tonic+self": "tonic_self"}
NOISE_MODES = (
    "absolute",
    "relative_global",
    "relative_frame",
    "percentile_global",
    "percentile_frame",
    "adaptive_frame",
    "median_mad_frame",
    "soft_relative_frame",
    "soft_adaptive_frame",
)


def _as_path(path: str | Path, label: str = "audio") -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} file not found: {result}")
    return result


def _resolve_fmin(fmin: str | float) -> float:
    if isinstance(fmin, str):
        try:
            return float(librosa.note_to_hz(fmin))
        except Exception as exc:
            raise ValueError("fmin must be a note name such as 'C1', or a frequency in Hz") from exc
    value = float(fmin)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("fmin must be positive")
    return value


def _effective_hop_length(sr: int, fps: float, hop_length: int) -> int:
    if float(fps) > 0:
        return max(1, int(round(int(sr) / float(fps))))
    return max(1, int(hop_length))


@lru_cache(maxsize=32)
def _cached_cqt(
    audio_path: str,
    file_mtime_ns: int,
    sr: int,
    hop_length: int,
    n_octaves: int,
    bins_per_octave: int,
    fmin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del file_mtime_ns
    y, loaded_sr = librosa.load(audio_path, sr=int(sr), mono=True)
    if y.size == 0:
        raise ValueError(f"audio is empty: {audio_path}")
    n_bins = int(n_octaves) * int(bins_per_octave)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"n_fft=.*is too large for input signal.*", category=UserWarning)
        cqt = librosa.cqt(
            y=y,
            sr=int(loaded_sr),
            hop_length=int(hop_length),
            n_bins=n_bins,
            bins_per_octave=int(bins_per_octave),
            fmin=float(fmin),
        )
    amplitude = np.asarray(np.abs(cqt), dtype=np.float64)
    pitch = librosa.hz_to_midi(float(fmin) * 2.0 ** (np.arange(n_bins, dtype=float) / int(bins_per_octave)))
    times = librosa.frames_to_time(np.arange(amplitude.shape[1]), sr=int(sr), hop_length=int(hop_length))
    return amplitude, np.asarray(pitch, dtype=float), np.asarray(times, dtype=float)


def _load_cqt(
    audio_path: str | Path,
    sr: int,
    hop_length: int,
    n_octaves: int,
    bins_per_octave: int,
    fmin: float,
    cache_cqt: bool,
) -> dict[str, Any]:
    path = _as_path(audio_path)
    key = (
        str(path),
        int(path.stat().st_mtime_ns),
        int(sr),
        int(hop_length),
        int(n_octaves),
        int(bins_per_octave),
        float(fmin),
    )
    if cache_cqt:
        amplitude, pitch, times = _cached_cqt(*key)
    else:
        amplitude, pitch, times = _cached_cqt.__wrapped__(*key)
    return {
        "audio_path": str(path),
        "cqt": amplitude,
        "pitch": pitch,
        "times": times,
        "sr": int(sr),
        "hop_length": int(hop_length),
        "n_octaves": int(n_octaves),
        "bins_per_octave": int(bins_per_octave),
        "fmin": float(fmin),
    }


def _context_window_frames(sr: int, hop_length: int, seconds: float) -> int:
    return max(1, int(int(sr) * float(seconds) / int(hop_length)))


def _previous_window_mean(matrix: np.ndarray, window_frames: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    result = np.zeros_like(matrix, dtype=float)
    if matrix.shape[1] <= 1:
        return result
    cumulative = np.cumsum(matrix, axis=1, dtype=float)
    result[:, 1:] = cumulative[:, :-1]
    if matrix.shape[1] > window_frames + 1:
        result[:, window_frames + 1 :] -= cumulative[:, : matrix.shape[1] - window_frames - 1]
    counts = np.minimum(np.arange(matrix.shape[1], dtype=float), float(window_frames))
    np.divide(result, counts[None, :], out=result, where=counts[None, :] > 0)
    return result


def _repeat_reference(
    source: np.ndarray,
    target_frames: int,
    reference_frame: int,
    average_frame: bool,
) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    if source.ndim != 2 or source.shape[1] == 0:
        raise ValueError("reference CQT is empty")
    if average_frame:
        frame = np.nanmean(source, axis=1)
    else:
        index = int(np.clip(reference_frame, 0, source.shape[1] - 1))
        frame = source[:, index]
    return np.repeat(frame[:, None], int(target_frames), axis=1)


def _reference_audio_path(tonic: str, tonal_chord_ref: bool, tonality: str = "major") -> Path:
    if tonic not in TONIC_OPTIONS:
        raise ValueError(f"reference_tonic must be one of {TONIC_OPTIONS}")
    tonality = str(tonality).strip().lower()
    if tonality not in TONALITY_OPTIONS:
        raise ValueError(f"reference_tonality must be one of {TONALITY_OPTIONS}")
    folder = "tonal_chords" if tonal_chord_ref else "notes"
    suffix = "maj" if tonality == "major" else "min"
    filename = f"{tonic}{suffix}.wav" if tonal_chord_ref else f"{tonic}4.wav"
    return REF_AUDIO_DIR / folder / filename


def _profile_to_key(profile: np.ndarray) -> tuple[str, str]:
    profile = np.nan_to_num(np.asarray(profile, dtype=float).reshape(12), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.sum(profile)) <= 0:
        return "C", "major"
    profile = np.maximum(profile, 0.0)
    profile /= max(float(np.linalg.norm(profile)), 1e-12)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    templates = {}
    for tonality, template in (("major", major), ("minor", minor)):
        centered = template - np.mean(template)
        templates[tonality] = centered / max(float(np.linalg.norm(centered)), 1e-12)
    centered_profile = profile - np.mean(profile)
    centered_profile /= max(float(np.linalg.norm(centered_profile)), 1e-12)
    candidates = [
        (float(centered_profile @ np.roll(template, root)), root, tonality)
        for root in range(12)
        for tonality, template in templates.items()
    ]
    best_score, best_root, best_tonality = max(candidates, key=lambda item: item[0])
    if not np.isfinite(best_score):
        return TONIC_OPTIONS[int(np.argmax(profile))], "major"
    return TONIC_OPTIONS[int(best_root)], str(best_tonality)


def _profile_to_tonic(profile: np.ndarray) -> str:
    """Backward-compatible root-only view of the key detector."""
    return _profile_to_key(profile)[0]


def _detect_tonic_segments(calc: dict[str, Any], window_seconds: float) -> list[dict[str, Any]]:
    cqt = np.asarray(calc["cqt"], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"n_fft=.*is too large for input signal.*", category=UserWarning)
        chroma = librosa.feature.chroma_cqt(
            C=cqt,
            sr=int(calc["sr"]),
            hop_length=int(calc["hop_length"]),
            fmin=float(calc["fmin"]),
            bins_per_octave=int(calc["bins_per_octave"]),
            n_chroma=12,
        )
    frame_count = cqt.shape[1]
    hop_seconds = float(calc["hop_length"]) / float(calc["sr"])
    region_frames = max(1, int(round(max(float(window_seconds), hop_seconds) / hop_seconds)))
    segments: list[dict[str, Any]] = []
    for start in range(0, frame_count, region_frames):
        end = min(frame_count, start + region_frames)
        profile = np.sum(np.maximum(chroma[:, start:min(end, chroma.shape[1])], 0.0), axis=1)
        tonic, tonality = _profile_to_key(profile)
        if segments and segments[-1]["tonic"] == tonic and segments[-1].get("tonality") == tonality:
            segments[-1]["end_frame"] = end
        else:
            segments.append({"start_frame": start, "end_frame": end, "tonic": tonic, "tonality": tonality})
    return segments


def _build_tonic_reference(
    calc: dict[str, Any],
    reference_tonic: str,
    reference_tonality: str,
    tonal_chord_ref: bool,
    auto_key_detect: bool,
    auto_key_window_seconds: float,
    reference_frame: int,
    average_frame: bool,
    cache_cqt: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target_frames = int(calc["cqt"].shape[1])
    if auto_key_detect:
        segments = _detect_tonic_segments(calc, auto_key_window_seconds)
    else:
        segments = [{
            "start_frame": 0,
            "end_frame": target_frames,
            "tonic": reference_tonic,
            "tonality": reference_tonality,
        }]
    result = np.zeros_like(calc["cqt"], dtype=float)
    frames: dict[str, np.ndarray] = {}
    keys = dict.fromkeys(
        (str(segment["tonic"]), str(segment.get("tonality", "major")))
        for segment in segments
    )
    for tonic, tonality in keys:
        frame_key = f"{tonic}:{tonality}"
        ref = _load_cqt(
            _reference_audio_path(tonic, tonal_chord_ref, tonality),
            calc["sr"],
            calc["hop_length"],
            calc["n_octaves"],
            calc["bins_per_octave"],
            calc["fmin"],
            cache_cqt,
        )["cqt"]
        frames[frame_key] = np.nanmean(ref, axis=1) if average_frame else ref[:, int(np.clip(reference_frame, 0, ref.shape[1] - 1))]
    for segment in segments:
        frame_key = f"{segment['tonic']}:{segment.get('tonality', 'major')}"
        result[:, segment["start_frame"] : segment["end_frame"]] = frames[frame_key][:, None]
    return result, segments


def _remove_noise_floor(
    matrix: np.ndarray,
    mode: str,
    noise_thr: float,
    noise_percentile: float,
    noise_mad_k: float,
    noise_soft: bool,
) -> np.ndarray:
    values = np.maximum(np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    mode = str(mode).lower()
    if mode.startswith("soft_"):
        noise_soft = True
        mode = mode.removeprefix("soft_")
    percentile = float(np.clip(noise_percentile, 0.0, 100.0))
    if mode == "absolute":
        floor: float | np.ndarray = float(noise_thr)
    elif mode == "relative_global":
        floor = float(np.max(values)) * float(noise_thr)
    elif mode == "relative_frame":
        floor = np.max(values, axis=0, keepdims=True) * float(noise_thr)
    elif mode == "percentile_global":
        nonzero = values[values > 1e-12]
        floor = float(np.percentile(nonzero, percentile)) if nonzero.size else 0.0
    elif mode == "percentile_frame":
        floor = np.percentile(values, percentile, axis=0, keepdims=True)
    elif mode == "adaptive_frame":
        floor = np.maximum(
            np.max(values, axis=0, keepdims=True) * float(noise_thr),
            np.percentile(values, percentile, axis=0, keepdims=True),
        )
    elif mode == "median_mad_frame":
        median = np.median(values, axis=0, keepdims=True)
        mad = np.median(np.abs(values - median), axis=0, keepdims=True)
        floor = median + float(noise_mad_k) * 1.4826 * mad
    else:
        raise ValueError(f"noise_filter_mode must be one of {NOISE_MODES}")
    return np.maximum(values - floor, 0.0) if noise_soft else np.where(values >= floor, values, 0.0)


def _safe_max(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray | float:
    maximum = np.max(np.abs(values), axis=axis, keepdims=keepdims)
    return np.where(maximum > 0, maximum, 1.0)


def _causal_window_max(matrix: np.ndarray, width: int) -> np.ndarray:
    values = np.abs(np.asarray(matrix, dtype=float))
    denominator = np.ones((1, values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        denominator[0, column] = max(float(np.max(values[:, max(0, column - width + 1) : column + 1])), 1.0e-12)
    return denominator


def _normalize_pair(
    calc: np.ndarray,
    ref: np.ndarray,
    mode: str,
    independent: bool,
    context_window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    calc = np.asarray(calc, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mode = str(mode).lower()
    if mode == "none":
        return calc.copy(), ref.copy()
    if mode == "frame":
        calc_den = _safe_max(calc, axis=0, keepdims=True)
        ref_den = _safe_max(ref, axis=0, keepdims=True)
    elif mode == "global":
        calc_den = _safe_max(calc)
        ref_den = _safe_max(ref)
    elif mode == "context_window":
        calc_den = _causal_window_max(calc, context_window_frames)
        ref_den = _causal_window_max(ref, context_window_frames)
    else:
        raise ValueError("normalize_mode must be 'frame', 'global', 'context_window', or 'none'")
    if independent:
        return calc / calc_den, ref / ref_den
    shared = np.maximum(calc_den, ref_den)
    return calc / shared, ref / shared


def _find_peak_indices(
    spectrum: np.ndarray,
    height: float | None,
    threshold: float | None,
    distance: float | None,
    prominence: float | None,
    width: float | None,
    top_n: int | None,
    normalize: bool,
) -> np.ndarray:
    original = np.asarray(spectrum, dtype=float)
    values = original / max(float(np.max(np.abs(original))), 1e-12) if normalize else original
    if values.size < 3:
        return np.array([int(np.argmax(values))], dtype=int)
    indices = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
    if height is not None:
        indices = indices[values[indices] >= float(height)]
    if threshold is not None and indices.size:
        indices = indices[
            (values[indices] - values[indices - 1] >= float(threshold))
            & (values[indices] - values[indices + 1] >= float(threshold))
        ]
    indices = indices[np.argsort(values[indices])[::-1]] if indices.size else indices
    if distance is not None and indices.size > 1:
        kept: list[int] = []
        for index in indices:
            if all(abs(int(index) - previous) >= int(distance) for previous in kept):
                kept.append(int(index))
        indices = np.asarray(kept, dtype=int)
    if prominence is not None and indices.size:
        left_min = np.minimum.accumulate(values)
        right_min = np.minimum.accumulate(values[::-1])[::-1]
        indices = indices[values[indices] - np.maximum(left_min[indices], right_min[indices]) >= float(prominence)]
    if width is not None and indices.size:
        kept_width = []
        for index in indices:
            half = values[index] / 2.0
            left = int(index)
            right = int(index)
            while left > 0 and values[left] >= half:
                left -= 1
            while right < values.size - 1 and values[right] >= half:
                right += 1
            if right - left >= float(width):
                kept_width.append(int(index))
        indices = np.asarray(kept_width, dtype=int)
    if top_n is not None:
        indices = indices[: int(top_n)]
    return indices if indices.size else np.array([int(np.argmax(values))], dtype=int)


def _replace_with_peaks(
    matrix: np.ndarray,
    height: float | None,
    threshold: float | None,
    distance: float | None,
    prominence: float | None,
    width: float | None,
    top_n: int | None,
    normalize: bool,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if all(value is None for value in (height, threshold, distance, prominence, width, top_n)) and not normalize:
        return matrix.copy()
    result = np.zeros_like(matrix, dtype=float)
    # The common sidebar configuration (prominence only) is vectorized over all frames.
    if distance is None and width is None and top_n is None and matrix.shape[0] >= 3:
        search = matrix / np.maximum(np.max(np.abs(matrix), axis=0, keepdims=True), 1e-12) if normalize else matrix
        mask = np.zeros_like(search, dtype=bool)
        mask[1:-1] = (search[1:-1] > search[:-2]) & (search[1:-1] >= search[2:])
        if height is not None:
            mask &= search >= float(height)
        if threshold is not None:
            neighbor_ok = np.zeros_like(mask)
            neighbor_ok[1:-1] = (
                (search[1:-1] - search[:-2] >= float(threshold))
                & (search[1:-1] - search[2:] >= float(threshold))
            )
            mask &= neighbor_ok
        if prominence is not None:
            left_min = np.minimum.accumulate(search, axis=0)
            right_min = np.minimum.accumulate(search[::-1], axis=0)[::-1]
            mask &= search - np.maximum(left_min, right_min) >= float(prominence)
        empty = ~np.any(mask, axis=0)
        if np.any(empty):
            mask[np.argmax(search[:, empty], axis=0), np.flatnonzero(empty)] = True
        result[mask] = matrix[mask]
        return result
    for column in range(matrix.shape[1]):
        indices = _find_peak_indices(matrix[:, column], height, threshold, distance, prominence, width, top_n, normalize)
        result[indices, column] = matrix[indices, column]
    return result


def _preprocess_pair(
    calc: np.ndarray,
    ref: np.ndarray,
    *,
    noise_filter: bool,
    noise_filter_mode: str,
    noise_thr: float,
    noise_percentile: float,
    noise_mad_k: float,
    noise_soft: bool,
    normalize_mode: str,
    normalize_independent: bool,
    normalize_context_window_frames: int,
    peak_height: float | None,
    peak_threshold: float | None,
    peak_distance: float | None,
    peak_prominence: float | None,
    peak_width: float | None,
    peak_top_n: int | None,
    peak_normalize: bool,
    repeated_reference: bool,
) -> tuple[np.ndarray, np.ndarray]:
    calc = np.array(calc, dtype=float, copy=True)
    ref = np.array(ref, dtype=float, copy=True)
    if noise_filter:
        kwargs = (noise_filter_mode, noise_thr, noise_percentile, noise_mad_k, noise_soft)
        calc = _remove_noise_floor(calc, *kwargs)
        ref = _remove_noise_floor(ref, *kwargs)
    calc, ref = _normalize_pair(calc, ref, normalize_mode, normalize_independent, normalize_context_window_frames)
    peak_args = (peak_height, peak_threshold, peak_distance, peak_prominence, peak_width, peak_top_n, peak_normalize)
    calc = _replace_with_peaks(calc, *peak_args)
    ref = _replace_with_peaks(ref, *peak_args)
    if repeated_reference and ref.shape[1] > 1:
        source_column = int(np.argmax(np.sum(np.abs(ref), axis=0)))
        ref = np.repeat(ref[:, source_column : source_column + 1], ref.shape[1], axis=1)
    return calc, ref


@lru_cache(maxsize=16)
def _rational_table(max_den: int) -> tuple[np.ndarray, np.ndarray]:
    fractions = sorted({Fraction(numerator, denominator) for denominator in range(1, int(max_den) + 1) for numerator in range(1, denominator + 1)})
    values = np.asarray([float(item) for item in fractions], dtype=float)
    complexity = np.asarray([np.log10(item.numerator * item.denominator) for item in fractions], dtype=float)
    return values, complexity


def _generate_dissonance_curve(
    k: int,
    bins_per_octave: int,
    max_den: int,
    err_mode: str,
    err_parameter: float,
    ignore_octave: bool,
) -> tuple[np.ndarray, np.ndarray]:
    relative_pitch = np.arange(-(k - 1), k, dtype=float) * (12.0 / int(bins_per_octave))
    delta = np.abs(relative_pitch)
    if ignore_octave:
        delta = np.mod(delta, 12.0)
    ratios = 2.0 ** (-delta / 12.0)
    rationals, complexity = _rational_table(int(max_den))
    values = np.empty(ratios.size, dtype=float)
    for start in range(0, ratios.size, 256):
        chunk = ratios[start : start + 256]
        distance = np.abs(chunk[:, None] - rationals[None, :])
        tolerance = float(err_parameter) if err_mode == "ratio_err" else chunk * float(err_parameter)
        nearby = distance < tolerance[:, None]
        scores = np.where(nearby, complexity[None, :], np.inf)
        selected = np.argmin(scores, axis=1)
        missing = ~np.any(nearby, axis=1)
        if np.any(missing):
            selected[missing] = np.argmin(distance[missing], axis=1)
        values[start : start + chunk.size] = np.round(complexity[selected], 3)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    values = (values - value_min) / (value_max - value_min) if value_max > value_min else np.zeros_like(values)
    return relative_pitch, values


def _curve_params_match(params: dict[str, Any], k: int, bins_per_octave: int, fmin_name: str | float, max_den: int, err_mode: str, err_parameter: float, ignore_octave: bool) -> bool:
    return (
        int(params.get("cqt_bins", -1)) == int(k)
        and int(params.get("bins_per_octave", -1)) == int(bins_per_octave)
        and int(params.get("max_den", -1)) == int(max_den)
        and str(params.get("err_mode", "")) == str(err_mode)
        and np.isclose(float(params.get("err_parameter", np.nan)), float(err_parameter))
        and bool(params.get("ignore_octave", False)) == bool(ignore_octave)
        and (not isinstance(fmin_name, str) or str(params.get("CQT_FMIN", "")) == fmin_name)
    )


def _load_curve_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    curve_file = path / "dissonance_curve.npz" if path.is_dir() else path
    params_file = path / "params.json" if path.is_dir() else path.with_name("params.json")
    if not curve_file.is_file():
        raise FileNotFoundError(f"dissonance curve not found: {curve_file}")
    with np.load(curve_file) as data:
        pitch = np.asarray(data["pitch"], dtype=float)
        dissonance = np.asarray(data["dissonance"], dtype=float)
    params = json.loads(params_file.read_text(encoding="utf-8")) if params_file.is_file() else {}
    return pitch, dissonance, params


def _resolve_curve_path(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.exists():
        raise FileNotFoundError(f"curve path not found: {result}")
    if not result.is_dir() and not result.is_file():
        raise ValueError(f"curve path is neither a file nor a directory: {result}")
    return result


def _curve_cache_name(k: int, bins_per_octave: int, fmin: str | float, max_den: int, err_mode: str, err_parameter: float, ignore_octave: bool) -> str:
    safe_fmin = str(fmin).replace("#", "sharp").replace(".", "p")
    return (
        f"maxden_{int(max_den)}__err_{err_mode}_{float(err_parameter):.4f}"
        f"__bpo_{int(bins_per_octave)}__bins_{int(k)}__fmin_{safe_fmin}__oct_{int(bool(ignore_octave))}"
    )


def _get_dissonance_curve(
    k: int,
    bins_per_octave: int,
    fmin: str | float,
    curve_path: str | Path | None,
    use_precomputed_curve: bool,
    save_generated_curve: bool,
    max_den: int,
    err_mode: str,
    err_parameter: float,
    ignore_octave: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if curve_path is not None:
        resolved_curve_path = _resolve_curve_path(curve_path)
        pitch, values, _ = _load_curve_bundle(resolved_curve_path)
        if values.size != 2 * k - 1:
            raise ValueError(f"curve length must be 2*K-1 ({2 * k - 1}), got {values.size}")
        return pitch, values, str(resolved_curve_path)
    if use_precomputed_curve and CURVE_DIR.is_dir():
        for params_file in CURVE_DIR.glob("*/params.json"):
            params = json.loads(params_file.read_text(encoding="utf-8"))
            if _curve_params_match(params, k, bins_per_octave, fmin, max_den, err_mode, err_parameter, ignore_octave):
                pitch, values, _ = _load_curve_bundle(params_file.parent)
                if values.size == 2 * k - 1:
                    return pitch, values, str(params_file.parent)
    pitch, values = _generate_dissonance_curve(k, bins_per_octave, max_den, err_mode, err_parameter, ignore_octave)
    source = "generated in memory"
    if save_generated_curve:
        folder = CURVE_DIR / _curve_cache_name(k, bins_per_octave, fmin, max_den, err_mode, err_parameter, ignore_octave)
        params = {
            "max_den": int(max_den),
            "err_mode": str(err_mode),
            "err_parameter": float(err_parameter),
            "ignore_octave": bool(ignore_octave),
            "n_octaves": int(k // bins_per_octave),
            "cqt_bins": int(k),
            "bins_per_octave": int(bins_per_octave),
            "CQT_FMIN": str(fmin),
        }
        try:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "dissonance_curve.npz", pitch=pitch, dissonance=values)
            (folder / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
            source = str(folder)
        except OSError:
            pass
    return pitch, values, source


def _matrix_convolution(
    calc: np.ndarray,
    ref: np.ndarray,
    curve: np.ndarray,
    use_torch: bool,
    device: str | None,
    dtype: Literal["float32", "float64"],
    batch_frames: int | None,
) -> tuple[np.ndarray, str]:
    k, frame_count = calc.shape
    if curve.size != 2 * k - 1:
        raise ValueError(f"curve length must be 2*K-1 ({2 * k - 1}), got {curve.size}")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if use_torch:
        try:
            import torch

            resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            torch_dtype = torch.float32 if dtype == "float32" else torch.float64
            curve_tensor = torch.as_tensor(curve, dtype=torch_dtype, device=resolved_device)
            # Each row is one valid correlation window. Reversing ref converts
            # correlation into a single dense matrix multiplication.
            windows = curve_tensor.unfold(0, k, 1)
            calc_tensor = torch.as_tensor(calc, dtype=torch_dtype, device=resolved_device)
            ref_tensor = torch.as_tensor(ref[::-1].copy(), dtype=torch_dtype, device=resolved_device)
            step = frame_count if not batch_frames else max(1, int(batch_frames))
            output = np.empty((k, frame_count), dtype=np.float32 if dtype == "float32" else np.float64)
            with torch.inference_mode():
                for start in range(0, frame_count, step):
                    end = min(frame_count, start + step)
                    ref_dissonance = windows @ ref_tensor[:, start:end]
                    batch = ref_dissonance * calc_tensor[:, start:end] / float(k)
                    output[:, start:end] = batch.detach().cpu().numpy()
            return output, str(resolved_device)
        except ImportError:
            pass
    np_dtype = np.float32 if dtype == "float32" else np.float64
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(curve, dtype=np_dtype), k)
    output = np.empty((k, frame_count), dtype=np_dtype)
    step = frame_count if not batch_frames else max(1, int(batch_frames))
    for start in range(0, frame_count, step):
        end = min(frame_count, start + step)
        ref_dissonance = windows @ np.asarray(ref[::-1, start:end], dtype=np_dtype)
        output[:, start:end] = ref_dissonance * np.asarray(calc[:, start:end], dtype=np_dtype) / float(k)
    return output, "numpy"


def calculate_dissonance_spectrum(
    audio_path: str | Path,
    *,
    reference_mode: Literal["specified", "tonic", "tonic_self", "tonic+self", "self", "context_window"] = "tonic+self",
    reference_audio_path: str | Path | None = None,
    reference_frame: int = 0,
    reference_tonic: str = "C",
    reference_tonality: Literal["major", "minor"] = "major",
    tonal_chord_ref: bool = False,
    auto_key_detect: bool = True,
    auto_key_window_seconds: float = 10.0,
    average_frame: bool = False,
    context_window_seconds: float = 1.0,
    sr: int = 22050,
    fps: float = 4.0,
    hop_length: int = 512,
    n_octaves: int = 8,
    bins_per_octave: int = 72,
    fmin: str | float = "C1",
    db_amp: bool = False,
    noise_filter: bool = True,
    noise_filter_mode: str = "relative_frame",
    noise_thr: float = 0.1,
    noise_percentile: float = 20.0,
    noise_mad_k: float = 2.5,
    noise_soft: bool = True,
    normalize_mode: Literal["frame", "global", "context_window", "none"] = "global",
    normalize_independent: bool = False,
    normalize_context_window_seconds: float = 1.0,
    peak_height: float | None = None,
    peak_threshold: float | None = None,
    peak_distance: float | None = None,
    peak_prominence: float | None = 0.015,
    peak_width: float | None = None,
    peak_top_n: int | None = None,
    peak_normalize: bool = False,
    curve_path: str | Path | None = None,
    curve_max_den: int = 60,
    curve_err_mode: Literal["ratio_err", "ratio_percentage"] = "ratio_percentage",
    curve_err_parameter: float = 0.01,
    curve_ignore_octave: bool = True,
    use_precomputed_curve: bool = True,
    save_generated_curve: bool = True,
    cache_cqt: bool = True,
    use_torch: bool = True,
    device: str | None = None,
    dtype: Literal["float32", "float64"] = "float32",
    batch_frames: int | None = None,
    return_details: bool = False,
) -> np.ndarray | dict[str, Any]:
    """Calculate the complete Visual Dissonance spectrum from an audio path.

    With defaults, the audio is split into 10-second key-detection regions;
    each detected tonic is combined with a self reference at 4 FPS. ``db_amp``
    is accepted for parity with the web sidebar; as in the web app, the final
    calculation always uses linear CQT amplitude.

    Returns a ``(pitch_bins, time_frames)`` NumPy matrix. Set
    ``return_details=True`` to also receive axes, processed CQT matrices and
    runtime metadata.
    """
    del db_amp
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")
    reference_mode = REFERENCE_MODE_ALIASES.get(reference_mode, reference_mode)
    if int(sr) <= 0 or int(n_octaves) <= 0 or int(bins_per_octave) <= 0:
        raise ValueError("sr, n_octaves and bins_per_octave must be positive")
    if curve_err_mode not in {"ratio_err", "ratio_percentage"}:
        raise ValueError("curve_err_mode must be 'ratio_err' or 'ratio_percentage'")
    if reference_tonality not in TONALITY_OPTIONS:
        raise ValueError(f"reference_tonality must be one of {TONALITY_OPTIONS}")
    effective_hop = _effective_hop_length(sr, fps, hop_length)
    fmin_hz = _resolve_fmin(fmin)
    calc_result = _load_cqt(audio_path, sr, effective_hop, n_octaves, bins_per_octave, fmin_hz, cache_cqt)
    calc_raw = np.asarray(calc_result["cqt"], dtype=float)
    frame_count = calc_raw.shape[1]
    segments: list[dict[str, Any]] = []

    if reference_mode == "specified":
        ref_path = reference_audio_path or (REF_AUDIO_DIR / "notes" / "C4.wav")
        ref_source = _load_cqt(ref_path, sr, effective_hop, n_octaves, bins_per_octave, fmin_hz, cache_cqt)["cqt"]
        ref_raw = _repeat_reference(ref_source, frame_count, reference_frame, average_frame)
        repeated_reference = True
    elif reference_mode in {"tonic", "tonic_self"}:
        ref_raw, segments = _build_tonic_reference(
            calc_result,
            reference_tonic,
            reference_tonality,
            tonal_chord_ref,
            auto_key_detect,
            auto_key_window_seconds,
            reference_frame,
            average_frame,
            cache_cqt,
        )
        repeated_reference = not auto_key_detect
    elif reference_mode == "context_window":
        ref_raw = _previous_window_mean(calc_raw, _context_window_frames(sr, effective_hop, context_window_seconds))
        repeated_reference = False
    elif average_frame:
        ref_raw = _repeat_reference(calc_raw, frame_count, reference_frame, True)
        repeated_reference = True
    else:
        ref_raw = calc_raw
        repeated_reference = False

    normalize_window_frames = _context_window_frames(sr, effective_hop, normalize_context_window_seconds)
    process_kwargs = dict(
        noise_filter=noise_filter,
        noise_filter_mode=noise_filter_mode,
        noise_thr=noise_thr,
        noise_percentile=noise_percentile,
        noise_mad_k=noise_mad_k,
        noise_soft=noise_soft,
        normalize_mode=normalize_mode,
        normalize_independent=normalize_independent,
        normalize_context_window_frames=normalize_window_frames,
        peak_height=peak_height,
        peak_threshold=peak_threshold,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        peak_width=peak_width,
        peak_top_n=peak_top_n,
        peak_normalize=peak_normalize,
    )
    calc_processed, ref_processed = _preprocess_pair(
        calc_raw,
        ref_raw,
        repeated_reference=repeated_reference,
        **process_kwargs,
    )
    curve_pitch, curve_values, curve_source = _get_dissonance_curve(
        calc_processed.shape[0],
        bins_per_octave,
        fmin,
        curve_path,
        use_precomputed_curve,
        save_generated_curve,
        curve_max_den,
        curve_err_mode,
        curve_err_parameter,
        curve_ignore_octave,
    )
    spectrum, backend = _matrix_convolution(
        calc_processed,
        ref_processed,
        curve_values,
        use_torch,
        device,
        dtype,
        batch_frames,
    )

    components: dict[str, np.ndarray] = {}
    if reference_mode == "tonic_self":
        self_ref = _repeat_reference(calc_raw, frame_count, reference_frame, True) if average_frame else calc_raw
        self_calc_processed, self_ref_processed = _preprocess_pair(
            calc_raw,
            self_ref,
            repeated_reference=average_frame,
            **process_kwargs,
        )
        self_spectrum, _ = _matrix_convolution(
            self_calc_processed,
            self_ref_processed,
            curve_values,
            use_torch,
            device,
            dtype,
            batch_frames,
        )
        components = {"tonic": spectrum, "self": self_spectrum}
        spectrum = spectrum + self_spectrum

    if not return_details:
        return spectrum
    return {
        "dissonance_spectrum": spectrum,
        "times": np.asarray(calc_result["times"], dtype=float),
        "pitch": np.asarray(calc_result["pitch"], dtype=float),
        "processed_calc_cqt": calc_processed,
        "processed_reference_cqt": ref_processed,
        "dissonance_curve_pitch": curve_pitch,
        "dissonance_curve": curve_values,
        "curve_source": curve_source,
        "backend": backend,
        "reference_mode": reference_mode,
        "tonic_segments": segments,
        "component_spectra": components,
        "sr": int(sr),
        "hop_length": int(effective_hop),
        "fps": float(sr) / int(effective_hop),
    }


def calculate_dissonance_intensity(
    dissonance_spectrum: np.ndarray,
    *,
    reduction: Literal["sum", "mean", "rms"] = "sum",
    nonnegative: bool = True,
    normalize: bool = False,
    pitch_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce a dissonance spectrum to one dissonance value per time frame."""
    spectrum = np.asarray(dissonance_spectrum, dtype=float)
    if spectrum.ndim != 2:
        raise ValueError("dissonance_spectrum must be a 2-D (pitch, time) matrix")
    values = np.maximum(spectrum, 0.0) if nonnegative else spectrum
    if pitch_weights is not None:
        weights = np.asarray(pitch_weights, dtype=float).reshape(-1)
        if weights.size != values.shape[0]:
            raise ValueError(f"pitch_weights length must equal pitch bins ({values.shape[0]})")
        values = values * weights[:, None]
    if reduction == "sum":
        intensity = np.sum(values, axis=0)
    elif reduction == "mean":
        intensity = np.mean(values, axis=0)
    elif reduction == "rms":
        intensity = np.sqrt(np.mean(np.square(values), axis=0))
    else:
        raise ValueError("reduction must be 'sum', 'mean', or 'rms'")
    if normalize and intensity.size:
        maximum = float(np.max(np.abs(intensity)))
        if maximum > 0:
            intensity = intensity / maximum
    return np.asarray(intensity, dtype=float)


# Short aliases for interactive/notebook use.
dissonance_spectrum = calculate_dissonance_spectrum
dissonance_intensity = calculate_dissonance_intensity


__all__ = [
    "calculate_dissonance_spectrum",
    "calculate_dissonance_intensity",
    "dissonance_spectrum",
    "dissonance_intensity",
]


def _main() -> None:
    """Run a small end-to-end smoke test with bundled reference audio."""
    import argparse

    parser = argparse.ArgumentParser(description="Visual Dissonance spectrum smoke test")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(REF_AUDIO_DIR / "notes" / "C4.wav"),
        help="Audio to analyze (default: bundled C4.wav)",
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Use NumPy instead of Torch for the matrix convolution",
    )
    args = parser.parse_args()

    # A deliberately small CQT keeps this executable example quick. Production
    # calls can simply omit these overrides to use the web app's 8x72 defaults.
    result = calculate_dissonance_spectrum(
        args.audio_path,
        reference_mode="self",
        n_octaves=2,
        bins_per_octave=12,
        fmin="C3",
        use_precomputed_curve=False,
        save_generated_curve=False,
        use_torch=not args.numpy,
        return_details=True,
    )
    spectrum = np.asarray(result["dissonance_spectrum"], dtype=float)
    intensity = calculate_dissonance_intensity(spectrum)

    print(f"audio: {Path(args.audio_path).resolve()}")
    print(f"dissonance spectrum: {spectrum.shape} (pitch x time)")
    print(f"dissonance intensity: {intensity.shape}")
    print(f"backend: {result['backend']}")
    print(f"maximum intensity: {float(np.max(intensity)) if intensity.size else 0.0:.6f}")


if __name__ == "__main__":
    _main()
