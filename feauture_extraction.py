"""Advanced audio feature extraction toolbox for DJ and music intelligence workflows.

This script focuses on extracting rich musical descriptors beyond basic tempo and key
analysis. It leverages optional best-in-class audio libraries (Essentia, madmom,
Spleeter/Demucs, OpenL3, pyloudnorm) when they are available, but keeps graceful
fallbacks that still expose valuable information using librosa and NumPy.

Example
=======
python feauture_extraction.py path/to/audio.wav --output analysis.json --embeddings

Highlights
==========
* Rhythm: onset curves, swing/groove, meter & polyrhythm hints, tempo evolution.
* Tonal: chroma, tonnetz, chord progression, tonal tension, modulation tracking.
* Spectral/Timbre: ZCR, spectral flatness, spectral centroid evolution.
* Dynamics: envelopes, crest factor, LUFS loudness (if pyloudnorm installed).
* Source & Content: optional stem separation, vocal activity, instrument proxies.
* Mood & Embeddings: Essentia MusicExtractor, OpenL3 embeddings, danceability etc.
* Spatial: stereo width, balance, phase coherence metrics.

The output is a JSON document that groups descriptors per musical facet together with
metadata explaining which libraries were used or which features were skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa


try:  # pragma: no cover - optional dependency
    import audioread  # type: ignore
    _AUDIOREAD_AVAILABLE = True
except ImportError:
    audioread = None
    _AUDIOREAD_AVAILABLE = False


DEFAULT_AUDIO_ROOT = Path("/Volumes/T7/AI_DJ/Music_Data")

try:
    import soundfile as sf  # type: ignore
    _SOUNDFILE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    sf = None
    _SOUNDFILE_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import pyloudnorm as pyln  # type: ignore
    _LOUDNESS_AVAILABLE = True
except ImportError:
    pyln = None
    _LOUDNESS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import openl3  # type: ignore
    _OPENL3_AVAILABLE = True
except ImportError:
    openl3 = None
    _OPENL3_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import essentia  # type: ignore
    import essentia.standard as es  # type: ignore
    _ESSENTIA_AVAILABLE = True
except ImportError:
    essentia = None
    es = None
    _ESSENTIA_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # type: ignore
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor  # type: ignore
    _MADMOM_AVAILABLE = True
except ImportError:
    DBNBeatTrackingProcessor = None
    RNNBeatProcessor = None
    DBNDownBeatTrackingProcessor = None
    RNNDownBeatProcessor = None
    _MADMOM_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from spleeter.separator import Separator  # type: ignore
    _SPLEETER_AVAILABLE = True
except ImportError:
    Separator = None
    _SPLEETER_AVAILABLE = False


class AudioLoadingError(RuntimeError):
    """Raised when an audio file cannot be decoded into a waveform."""


@dataclass
class FeatureConfig:
    """User-facing knobs for heavy computations and sampling."""

    sample_rate: Optional[int] = None
    hop_length: int = 512
    enable_stems: bool = False
    enable_embeddings: bool = False
    disable_essentia: bool = False
    disable_madmom: bool = False
    max_array_points: int = 512

    @property
    def use_essentia(self) -> bool:
        return _ESSENTIA_AVAILABLE and not self.disable_essentia

    @property
    def use_madmom(self) -> bool:
        return _MADMOM_AVAILABLE and not self.disable_madmom


###############################################################################
# Utility helpers
###############################################################################

def _downsample_array(values: np.ndarray, max_points: int) -> List[float]:
    """Return a down-sampled list representation to keep JSON readable."""

    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    if arr.size <= max_points:
        return arr.tolist()
    indices = np.linspace(0, arr.size - 1, max_points).astype(int)
    return arr[indices].tolist()


def _stats(values: np.ndarray) -> Dict[str, float]:
    """Summarise an array with descriptive statistics."""

    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _log_availability(flag: bool, name: str) -> str:
    return "available" if flag else f"missing ({name} not installed)"


def _spectral_flux_onset(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Lightweight onset envelope that avoids librosa's mel dependency."""

    magnitude = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=hop_length))
    if magnitude.size == 0:
        return np.zeros(0, dtype=float)
    energy = np.sum(magnitude, axis=0)
    if energy.size == 0:
        return np.zeros(0, dtype=float)
    diff = np.diff(energy, prepend=energy[0])
    diff = np.maximum(diff, 0.0)
    max_val = np.max(diff)
    if max_val > 0:
        diff = diff / max_val
    return diff.astype(float)


def load_audio(path: Path, config: FeatureConfig) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load mono and stereo representations of the audio signal."""

    try:
        y_stereo, sr = librosa.load(path, sr=config.sample_rate, mono=False)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        message = f"Unable to load audio from {path}: {exc}"
        hints = []
        if _SOUNDFILE_AVAILABLE and isinstance(exc, sf.LibsndfileError):
            hints.append("Libsndfile could not decode the file; ensure it is a valid audio asset or convert it to WAV/FLAC.")
        if _AUDIOREAD_AVAILABLE and isinstance(exc, audioread.NoBackendError):
            hints.append("No MP3 decoding backend is available. Install FFmpeg/GStreamer (e.g. `sudo apt-get install ffmpeg`) to enable MP3 decoding.")
        if hints:
            message = f"{message} {' '.join(hints)}"
        raise AudioLoadingError(message) from exc

    if y_stereo.ndim == 1:
        y_mono = y_stereo
        y_stereo = np.expand_dims(y_stereo, axis=0)
    else:
        y_mono = np.mean(y_stereo, axis=0)
    return y_mono, y_stereo, sr


def _estimate_swing(beat_times: np.ndarray) -> Optional[Dict[str, float]]:
    if beat_times.size < 4:
        return None
    intervals = np.diff(beat_times)
    if intervals.size < 2:
        return None
    even = intervals[::2]
    odd = intervals[1::2]
    if even.size == 0 or odd.size == 0:
        return None
    swing_ratio = float(np.mean(odd) / (np.mean(even) + 1e-9))
    micro_timing = float(np.std(intervals - np.mean(intervals)))
    return {
        "swing_ratio": swing_ratio,
        "micro_timing_std": micro_timing,
    }


def _tempo_variation(onset_env: np.ndarray, sr: int, hop_length: int) -> Dict[str, float]:
    tempi = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None)
    if tempi.size == 0:
        return {"median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return _stats(tempi)


def _polyrhythm_candidates(onset_env: np.ndarray, sr: int, hop_length: int) -> Dict[str, Any]:
    if onset_env.size == 0:
        return {"primary_tempi": [], "relative_ratios": []}
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo_axis = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
    strength = np.mean(tempogram, axis=1)
    finite_mask = np.isfinite(tempo_axis)
    tempo_axis = tempo_axis[finite_mask]
    strength = strength[finite_mask]
    order = np.argsort(strength)[-3:][::-1]
    candidates = tempo_axis[order]
    ratios = []
    if candidates.size >= 2:
        base = candidates[0]
        for tempo in candidates[1:]:
            if tempo == 0:
                continue
            ratios.append(float(base / tempo))
    return {
        "primary_tempi": candidates.tolist(),
        "relative_ratios": ratios,
    }


def _danceability_score(tempo_bpm: float, onset_env: np.ndarray) -> float:
    if tempo_bpm <= 0 or onset_env.size == 0:
        return 0.0
    preferred = 120.0
    width = 40.0
    tempo_component = math.exp(-((tempo_bpm - preferred) ** 2) / (2 * width ** 2))
    onset_norm = onset_env / (np.max(onset_env) + 1e-9)
    rhythmic_variation = float(np.std(onset_norm))
    combo = 0.6 * tempo_component + 0.4 * np.clip(rhythmic_variation, 0.0, 1.0)
    return float(np.clip(combo, 0.0, 1.0))


def _estimate_chord_progression(chroma: np.ndarray, sr: int, hop_length: int) -> Dict[str, Any]:
    if chroma.size == 0:
        return {"progression": [], "histogram": {}}

    chroma = np.asarray(chroma, dtype=float)
    chroma = np.maximum(chroma, 0.0)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma_norm = chroma / norms

    scores = CHORD_TEMPLATES @ chroma_norm
    best_indices = np.argmax(scores, axis=0)
    best_scores = scores[best_indices, np.arange(scores.shape[1])]
    second_scores = np.partition(scores, -2, axis=0)[-2, :]
    confidence = best_scores - second_scores

    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    progression: List[Dict[str, Any]] = []
    chord_hist: Dict[str, int] = {}

    start_idx = 0
    current_idx = int(best_indices[0])
    for frame in range(1, best_indices.size):
        idx = int(best_indices[frame])
        if idx != current_idx:
            chord = CHORD_LABELS[current_idx]
            chord_hist[chord] = chord_hist.get(chord, 0) + 1
            segment_conf = float(np.mean(confidence[start_idx:frame]))
            progression.append(
                {
                    "chord": chord,
                    "start_sec": float(times[start_idx]),
                    "end_sec": float(times[frame]),
                    "confidence": segment_conf,
                }
            )
            start_idx = frame
            current_idx = idx

    chord = CHORD_LABELS[current_idx]
    chord_hist[chord] = chord_hist.get(chord, 0) + 1
    segment_conf = float(np.mean(confidence[start_idx:]))
    progression.append(
        {
            "chord": chord,
            "start_sec": float(times[start_idx]),
            "end_sec": float(times[-1]),
            "confidence": segment_conf,
        }
    )

    return {
        "progression": progression,
        "histogram": chord_hist,
    }


def _madmom_meter(path: Path) -> Optional[Dict[str, Any]]:
    if not _MADMOM_AVAILABLE:
        return None
    if RNNDownBeatProcessor is None or DBNDownBeatTrackingProcessor is None:
        return None
    rnn = RNNDownBeatProcessor()
    activation = rnn(str(path))
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4, 6], fps=100)
    downbeats = dbn(activation)
    meters, counts = np.unique(downbeats[:, 1].astype(int), return_counts=True)
    meter_estimate = int(meters[np.argmax(counts)]) if meters.size else None
    return {
        "downbeat_count": int(downbeats.shape[0]) if downbeats.size else 0,
        "meter_estimate": meter_estimate,
    }


def extract_rhythm_features(path: Path, y: np.ndarray, sr: int, config: FeatureConfig) -> Dict[str, Any]:
    onset_env = _spectral_flux_onset(y, config.hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=config.hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=config.hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=config.hop_length)

    madmom_info = _madmom_meter(path) if config.use_madmom else None

    features: Dict[str, Any] = {
        "library": "librosa",
        "onset_curve_summary": {
            "stats": _stats(onset_env),
            "times": _downsample_array(times, config.max_array_points),
            "curve": _downsample_array(onset_env, config.max_array_points),
        },
        "tempo_bpm_global": float(tempo),
        "beat_times_sec": _downsample_array(beat_times, config.max_array_points),
        "tempo_variation_bpm": _tempo_variation(onset_env, sr, config.hop_length),
        "polyrhythm_candidates": _polyrhythm_candidates(onset_env, sr, config.hop_length),
        "madmom_downbeat": (
            madmom_info
            if madmom_info is not None
            else ({"status": "madmom_no_detection"} if config.use_madmom else {"status": _log_availability(False, "madmom")})
        ),
        "danceability_score": _danceability_score(float(tempo), onset_env),
        "onset_curve_times": _downsample_array(times, config.max_array_points),
        "onset_curve_values": _downsample_array(onset_env, config.max_array_points),
    }

    swing = _estimate_swing(beat_times)
    if swing:
        features["swing_analysis"] = swing
        groove_conf = float(np.clip(swing.get("micro_timing_std", 0.0) * 2.0, 0.0, 1.0))
        features["groove_confidence"] = groove_conf
    else:
        features["swing_analysis"] = {"status": "not_enough_beats"}
        features["groove_confidence"] = 0.0

    return features


###############################################################################
# Tonal & spectral descriptors
###############################################################################

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_chord_templates() -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    templates: List[np.ndarray] = []
    for idx, pitch in enumerate(PITCH_CLASSES):
        major = np.zeros(12, dtype=float)
        major[[idx, (idx + 4) % 12, (idx + 7) % 12]] = 1.0
        minor = np.zeros(12, dtype=float)
        minor[[idx, (idx + 3) % 12, (idx + 7) % 12]] = 1.0
        labels.append(f"{pitch}:maj")
        labels.append(f"{pitch}:min")
        templates.append(major / np.linalg.norm(major))
        templates.append(minor / np.linalg.norm(minor))
    return labels, np.stack(templates, axis=0)


CHORD_LABELS, CHORD_TEMPLATES = _build_chord_templates()


def _estimate_key(chroma: np.ndarray) -> Dict[str, Any]:
    chroma = np.asarray(chroma)
    if chroma.ndim == 1:
        chroma_mean = chroma
    else:
        chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (np.max(chroma_mean) + 1e-9)

    key_scores: Dict[str, float] = {}
    for shift, pitch in enumerate(PITCH_CLASSES):
        rotated = np.roll(chroma_mean, -shift)
        key_scores[f"{pitch} major"] = float(np.dot(MAJOR_PROFILE, rotated))
        key_scores[f"{pitch} minor"] = float(np.dot(MINOR_PROFILE, rotated))

    best_key, best_score = max(key_scores.items(), key=lambda item: item[1])
    total_score = sum(key_scores.values()) + 1e-9
    return {
        "key": best_key,
        "confidence": float(best_score / total_score),
        "all_scores": {k: float(v) for k, v in key_scores.items()},
    }


def _tonal_tension(tonnetz: np.ndarray) -> Dict[str, float]:
    derivative = np.gradient(tonnetz, axis=1)
    tension = np.linalg.norm(derivative, axis=0)
    return _stats(tension)


def _modulation_track(chroma: np.ndarray, sr: int, hop_length: int) -> Dict[str, Any]:
    num_frames = chroma.shape[1]
    if num_frames == 0:
        return {"times_sec": [], "keys": [], "confidence": []}
    step = max(1, num_frames // 256)
    sampled_indices = np.arange(0, num_frames, step)
    times = librosa.frames_to_time(sampled_indices, sr=sr, hop_length=hop_length)
    frame_keys: List[str] = []
    key_conf: List[float] = []
    for idx in sampled_indices:
        frame = chroma[:, idx]
        key_info = _estimate_key(frame)
        frame_keys.append(key_info["key"])
        key_conf.append(key_info["confidence"])
    return {
        "times_sec": times.tolist(),
        "keys": frame_keys,
        "confidence": key_conf,
    }


def _dissonance_index(y: np.ndarray, sr: int) -> float:
    # Simple roughness proxy via spectral spread and flatness interplay.
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spectral_flatness = librosa.feature.spectral_flatness(S=stft)
    spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    roughness = float(np.mean(spectral_flatness) * np.mean(spectral_contrast))
    return roughness


def extract_tonal_features(y: np.ndarray, sr: int, config: FeatureConfig) -> Dict[str, Any]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    zcr = librosa.feature.zero_crossing_rate(y)
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    tonal: Dict[str, Any] = {
        "chroma_stats": _stats(chroma),
        "tonnetz_stats": _stats(tonnetz),
        "tonal_tension": _tonal_tension(tonnetz),
        "zero_crossing_rate": _stats(zcr),
        "spectral_flatness": _stats(spec_flatness),
        "spectral_centroid": _stats(spec_centroid),
        "dissonance_index": _dissonance_index(y, sr),
    }

    tonal.update(_estimate_key(chroma))
    tonal["modulation"] = _modulation_track(chroma, sr, config.hop_length)
    tonal["chords"] = _estimate_chord_progression(chroma, sr, config.hop_length)

    if config.use_essentia:
        tonal["essentia_status"] = "available"
        tonal.update(extract_essentia_tonal(y, sr))
    else:
        tonal["essentia_status"] = _log_availability(config.use_essentia, "Essentia")

    return tonal


###############################################################################
# Dynamics & loudness
###############################################################################

def _attack_decay_envelope(y: np.ndarray, sr: int, hop_length: int) -> Dict[str, float]:
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length).ravel()
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    onset_envelope = _spectral_flux_onset(y, hop_length)

    attack_time = 0.0
    if onset_envelope.size:
        threshold = np.max(onset_envelope) * 0.7
        crossings = np.where(onset_envelope >= threshold)[0]
        if crossings.size:
            attack_time = float(times[min(crossings[0], times.size - 1)])

    decay_time = 0.0
    if rms.size:
        peak_idx = int(np.argmax(rms))
        decay_segment = rms[peak_idx:]
        if decay_segment.size:
            decay_threshold = np.max(decay_segment) * 0.5
            below = np.where(decay_segment <= decay_threshold)[0]
            if below.size:
                decay_time = float(below[0] * hop_length / sr)

    return {
        "attack_time_sec": attack_time,
        "decay_time_sec": decay_time,
    }


def _crest_factor(y: np.ndarray) -> float:
    peak = np.max(np.abs(y))
    rms = math.sqrt(np.mean(np.square(y))) + 1e-9
    return float(peak / rms)


def _lufs(y: np.ndarray, sr: int) -> Optional[float]:
    if not _LOUDNESS_AVAILABLE:
        return None
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    return float(loudness)


def extract_dynamic_features(y: np.ndarray, sr: int, config: FeatureConfig) -> Dict[str, Any]:
    dynamics: Dict[str, Any] = {
        "rms": _stats(librosa.feature.rms(y=y, hop_length=config.hop_length)),
        "attack_decay": _attack_decay_envelope(y, sr, config.hop_length),
        "crest_factor": _crest_factor(y),
    }

    loudness = _lufs(y, sr)
    if loudness is None:
        dynamics["lufs"] = {"status": _log_availability(False, "pyloudnorm")}
    else:
        dynamics["lufs"] = loudness

    return dynamics


###############################################################################
# Spatial descriptors
###############################################################################

def extract_spatial_features(y_stereo: np.ndarray, sr: int) -> Dict[str, Any]:
    if y_stereo.shape[0] < 2:
        return {"status": "mono_signal"}
    left, right = y_stereo[0], y_stereo[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    def rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(signal))) + 1e-9)

    stft_left = librosa.stft(left)
    stft_right = librosa.stft(right)
    phase_diff = np.angle(stft_left) - np.angle(stft_right)
    coherence = float(np.mean(np.cos(phase_diff)))

    return {
        "stereo_width": float(np.mean(np.abs(side)) / (np.mean(np.abs(mid)) + 1e-9)),
        "balance_db": 20 * math.log10(rms(left) / (rms(right) + 1e-9)),
        "phase_coherence": coherence,
        "side_mid_ratio_db": 20 * math.log10((rms(side) + 1e-9) / (rms(mid) + 1e-9)),
    }


###############################################################################
# Optional: Essentia boosted descriptors
###############################################################################

def extract_essentia_tonal(y: np.ndarray, sr: int) -> Dict[str, Any]:
    if not _ESSENTIA_AVAILABLE:
        return {"status": "Essentia_not_available"}

    array = essentia.array(y.astype(np.float32))
    tonal_extractor = es.TonalExtractor(sampleRate=sr)
    tonal = tonal_extractor(array)
    chords_strength = tonal.get("chords_strength")
    if hasattr(chords_strength, "tolist"):
        chords_strength_value = float(np.mean(np.array(chords_strength)))
    else:
        chords_strength_value = float(chords_strength) if isinstance(chords_strength, (int, float)) else None
    chords_histogram = tonal.get("chords_histogram")
    if hasattr(chords_histogram, "tolist"):
        chords_histogram_list = chords_histogram.tolist()
    else:
        chords_histogram_list = chords_histogram
    return {
        "essentia_key": f"{tonal.get('key', '')} {tonal.get('scale', '')}".strip(),
        "essentia_key_strength": float(tonal.get("strength", 0.0)),
        "chords_progression": tonal.get("chords_progression"),
        "chords_histogram": chords_histogram_list,
        "chords_strength_mean": chords_strength_value,
    }


def extract_essentia_highlevel(path: Path) -> Dict[str, Any]:
    if not _ESSENTIA_AVAILABLE:
        return {}
    music_extractor = es.MusicExtractor(
        lowlevelStats=True,
        rhythmStats=True,
        tonalStats=True,
        highlevelStats=True,
    )
    result = music_extractor(str(path))

    highlevel = result.get("highlevel", {})
    mood = {k: v.get("value") for k, v in highlevel.items() if isinstance(v, dict) and "mood" in k}

    genre_info = highlevel.get("genre_dortmund")
    if isinstance(genre_info, dict):
        probabilities = genre_info.get("probability")
        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()
        genre_data = {
            "predicted": genre_info.get("value"),
            "probabilities": probabilities,
            "labels": genre_info.get("all", {}).get("labels") if isinstance(genre_info.get("all"), dict) else None,
        }
    else:
        genre_data = None

    voice_info = highlevel.get("voice_instrumental")
    if isinstance(voice_info, dict):
        voice_data = {}
        for key in ("value", "probability"):
            val = voice_info.get(key)
            if hasattr(val, "tolist"):
                voice_data[key] = val.tolist()
            else:
                voice_data[key] = val
    else:
        voice_data = None

    def _serialize(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _serialize(v) for k, v in value.items()}
        return value

    lowlevel = result.get("lowlevel", {})
    timbre_keys = ["mfcc", "gfcc", "spectral_contrast", "spectral_complexity"]
    timbre_summary = {}
    for key in timbre_keys:
        entry = lowlevel.get(key)
        if entry is not None:
            timbre_summary[key] = _serialize(entry)

    return {
        "danceability": highlevel.get("danceability", {}).get("value"),
        "mood": mood,
        "genre": genre_data,
        "voice_instrumental": voice_data,
        "timbre_descriptors": timbre_summary,
    }


def extract_vocal_activity(y: np.ndarray, sr: int) -> Dict[str, Any]:
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": f"pyin_failed: {exc}"}

    if voiced_flag is None:
        voiced_ratio = 0.0
    else:
        voiced_flag = np.nan_to_num(voiced_flag.astype(float), nan=0.0)
        voiced_ratio = float(np.mean(voiced_flag))

    if f0 is not None:
        mean_val = np.nanmean(f0)
        std_val = np.nanstd(f0)
        pitch_mean = float(mean_val) if np.isfinite(mean_val) else None
        pitch_std = float(std_val) if np.isfinite(std_val) else None
    else:
        pitch_mean = None
        pitch_std = None

    if voiced_prob is not None:
        prob_mean = np.nanmean(voiced_prob)
        voiced_probability_mean = float(prob_mean) if np.isfinite(prob_mean) else None
    else:
        voiced_probability_mean = None

    return {
        "voiced_ratio": voiced_ratio,
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "voiced_probability_mean": voiced_probability_mean,
    }


def extract_embeddings(path: Path, config: FeatureConfig) -> Dict[str, Any]:
    embeddings: Dict[str, Any] = {}
    if config.enable_embeddings and _OPENL3_AVAILABLE:
        audio, sr = librosa.load(path, sr=config.sample_rate, mono=False)
        if audio.ndim == 1:
            waveform = audio.astype(np.float32)
        else:
            waveform = audio.T.astype(np.float32)
        try:
            embs, timestamps = openl3.get_audio_embedding(waveform, sr, hop_size=1.0)
        except Exception as exc:  # pragma: no cover - defensive
            embeddings["openl3"] = {"status": f"openl3_failed: {exc}"}
        else:
            embeddings["openl3"] = {
                "shape": list(embs.shape),
                "embedding_mean": _downsample_array(np.mean(embs, axis=0), config.max_array_points),
                "timestamps": timestamps.tolist(),
            }
    elif config.enable_embeddings:
        embeddings["openl3"] = {"status": _log_availability(False, "openl3")}
    return embeddings


def extract_stems(path: Path, config: FeatureConfig) -> Dict[str, Any]:
    if not config.enable_stems:
        return {"status": "stems_disabled"}
    if not _SPLEETER_AVAILABLE:
        return {"status": _log_availability(False, "Spleeter")}
    separator = Separator("spleeter:4stems")
    audio, sr = librosa.load(path, sr=config.sample_rate, mono=False)
    if audio.ndim == 1:
        waveform = np.expand_dims(audio.astype(np.float32), axis=1)
    else:
        waveform = audio.T.astype(np.float32)
    try:
        stems = separator.separate(waveform)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": f"spleeter_failed: {exc}"}
    stats: Dict[str, Any] = {}
    for stem_name, stem_audio in stems.items():
        if stem_audio.ndim == 2:
            mono = stem_audio.mean(axis=1)
        else:
            mono = stem_audio
        stats[stem_name] = {
            "rms": _stats(librosa.feature.rms(y=mono)),
            "spectral_centroid": _stats(librosa.feature.spectral_centroid(y=mono, sr=sr)),
        }
    return stats


###############################################################################
# Instrument & mood heuristics
###############################################################################


def _instrument_presence(y: np.ndarray, sr: int) -> Dict[str, float]:
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    if stft.size == 0:
        return {
            "bass_presence": 0.0,
            "mid_presence": 0.0,
            "treble_presence": 0.0,
            "percussive_ratio": 0.0,
        }
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(np.mean(stft) + 1e-9)
    bass = float(np.mean(stft[freqs < 150.0])) / total
    mid = float(np.mean(stft[(freqs >= 150.0) & (freqs < 2000.0)])) / total
    treble = float(np.mean(stft[freqs >= 2000.0])) / total

    harmonic, percussive = librosa.decompose.hpss(stft)
    percussive_ratio = float(np.mean(percussive) / (np.mean(percussive) + np.mean(harmonic) + 1e-9))

    return {
        "bass_presence": float(np.clip(bass, 0.0, 1.0)),
        "mid_presence": float(np.clip(mid, 0.0, 1.0)),
        "treble_presence": float(np.clip(treble, 0.0, 1.0)),
        "percussive_ratio": float(np.clip(percussive_ratio, 0.0, 1.0)),
    }


def _instrument_summary(y: np.ndarray, sr: int, vocal_features: Dict[str, Any]) -> Dict[str, Any]:
    presence = _instrument_presence(y, sr)
    voiced_ratio = vocal_features.get("voiced_ratio", 0.0) or 0.0
    vocal_mean_prob = vocal_features.get("voiced_probability_mean")

    return {
        **presence,
        "vocal_presence_proxy": float(np.clip(voiced_ratio, 0.0, 1.0)),
        "vocal_probability_mean": float(np.clip(vocal_mean_prob, 0.0, 1.0)) if vocal_mean_prob is not None else None,
        "likely_bass_heavy": presence["bass_presence"] > 0.35,
        "likely_bright_timbre": presence["treble_presence"] > 0.35,
    }


def _normalise(value: float, min_val: float, max_val: float) -> float:
    if np.isnan(value):
        return 0.0
    if max_val == min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def _mood_profile(rhythm: Dict[str, Any], tonal: Dict[str, Any], dynamics: Dict[str, Any]) -> Dict[str, Any]:
    tempo = float(rhythm.get("tempo_bpm_global", 0.0))
    danceability = float(rhythm.get("danceability_score", 0.0))
    centroid_mean = float(tonal.get("spectral_centroid", {}).get("mean", 0.0))
    flatness_mean = float(tonal.get("spectral_flatness", {}).get("mean", 0.0))
    dissonance = float(tonal.get("dissonance_index", 0.0))
    rms_mean = float(dynamics.get("rms", {}).get("mean", 0.0))

    energy = 0.6 * _normalise(rms_mean, 0.01, 0.3) + 0.4 * _normalise(tempo, 60.0, 180.0)
    brightness = _normalise(centroid_mean, 1000.0, 6000.0)
    roughness = _normalise(dissonance, 0.0, 15.0)
    valence = np.clip(0.5 * brightness + 0.3 * danceability - 0.2 * roughness, 0.0, 1.0)
    arousal = np.clip(0.6 * energy + 0.4 * (1.0 - flatness_mean), 0.0, 1.0)

    mood_state = "energetic" if arousal >= 0.6 else "chill"
    if valence >= 0.6:
        mood_state += " & positive"
    elif valence <= 0.4:
        mood_state += " & moody"

    return {
        "danceability": danceability,
        "valence": float(valence),
        "arousal": float(arousal),
        "energy": float(np.clip(energy, 0.0, 1.0)),
        "descriptor": mood_state,
    }


###############################################################################
# Master extraction
###############################################################################

def analyze_audio(path: Path, config: FeatureConfig) -> Dict[str, Any]:
    y_mono, y_stereo, sr = load_audio(path, config)

    metadata = {
        "path": str(path),
        "sample_rate": sr,
        "duration_sec": float(len(y_mono) / sr),
        "dependencies": {
            "essentia": config.use_essentia,
            "madmom": config.use_madmom,
            "pyloudnorm": _LOUDNESS_AVAILABLE,
            "openl3": _OPENL3_AVAILABLE,
            "spleeter": _SPLEETER_AVAILABLE,
        },
    }

    analysis: Dict[str, Any] = {
        "rhythm": extract_rhythm_features(path, y_mono, sr, config),
        "tonal": extract_tonal_features(y_mono, sr, config),
        "dynamics": extract_dynamic_features(y_mono, sr, config),
        "spatial": extract_spatial_features(y_stereo, sr),
        "vocal_activity": extract_vocal_activity(y_mono, sr),
        "stems": extract_stems(path, config),
        "embeddings": extract_embeddings(path, config),
    }

    analysis["instrumentation"] = _instrument_summary(y_mono, sr, analysis["vocal_activity"])
    analysis["mood"] = _mood_profile(analysis["rhythm"], analysis["tonal"], analysis["dynamics"])

    if config.use_essentia:
        analysis["essentia_highlevel"] = extract_essentia_highlevel(path)

    return {
        "metadata": metadata,
        "analysis": analysis,
    }


###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced audio feature extraction for DJ intelligence.")
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        help="Path to an audio file (wav/mp3/flac etc.). Optional if Music_Data default is available.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON file destination.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Target sample rate for analysis (default: native).")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for frame-based features.")
    parser.add_argument("--enable-stems", action="store_true", help="Run stem separation (requires Spleeter/Demucs).")
    parser.add_argument("--embeddings", action="store_true", help="Compute OpenL3 audio embeddings if available.")
    parser.add_argument("--disable-essentia", action="store_true", help="Skip Essentia-based descriptors even if installed.")
    parser.add_argument("--disable-madmom", action="store_true", help="Skip madmom-based meter detection even if installed.")
    parser.add_argument("--max-array-points", type=int, default=512, help="Maximum points stored per time-series in JSON output.")
    parser.add_argument(
        "--batch-folder",
        type=Path,
        help=(
            "Process every audio file in this folder (pattern *.mp3/*.wav). "
            "If set, the positional audio argument is ignored. Defaults to Music_Data when omitted."
        ),
    )
    parser.add_argument(
        "--batch-pattern",
        default="*.mp3",
        help="Glob pattern for files when using --batch-folder (use '**/*.mp3' for nested folders).",
    )
    parser.add_argument(
        "--batch-out",
        type=Path,
        default=None,
        help="Optional directory for batch outputs (default: alongside each audio file).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute analysis even if the JSON output already exists.",
    )
    args = parser.parse_args()

    if args.audio is None and args.batch_folder is None:
        if DEFAULT_AUDIO_ROOT.exists():
            args.batch_folder = DEFAULT_AUDIO_ROOT
            default_pattern = parser.get_default("batch_pattern")
            if args.batch_pattern == default_pattern:
                args.batch_pattern = "**/*.mp3"
        else:
            parser.error("An audio file or --batch-folder must be provided.")

    return args


def _destination_for(path: Path, output: Optional[Path]) -> Path:
    if output and str(output) == "-":
        raise ValueError("stdout destination should be handled before calling _destination_for")
    return output if output else path.with_suffix(".json")


def _run_single(path: Path, config: FeatureConfig, output: Optional[Path], overwrite: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if output and str(output) == "-":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_audio(path, config)
        print(json.dumps(result, indent=2))
        return

    destination = _destination_for(path, output)
    if destination.exists() and not overwrite:
        print(f"Skipping {path} (analysis already exists at {destination}; use --overwrite to recompute)")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = analyze_audio(path, config)
        except AudioLoadingError as exc:
            message = f"Failed to analyze {path}: {exc}"
            print(message)
            error_payload = {"metadata": {"path": str(path)}, "error": {"type": "AudioLoadingError", "detail": str(exc)}}
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(error_payload, indent=2))
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))


def _run_batch(args: argparse.Namespace, config: FeatureConfig) -> None:
    folder = args.batch_folder
    pattern = args.batch_pattern
    out_dir = args.batch_out
    matched = sorted(folder.glob(pattern)) if folder else []
    if not matched:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    for path in matched:
        if path.name.startswith("._"):
            print(f"Skipping resource-fork placeholder {path}")
            continue
        if out_dir is not None:
            try:
                relative_path = path.relative_to(folder)
            except ValueError:
                relative_path = Path(path.name)
            output_path = out_dir / relative_path.with_suffix(".json")
        else:
            output_path = path.with_suffix(".json")
        if output_path.exists() and not args.overwrite:
            print(f"Skipping {path} (analysis already exists at {output_path}; use --overwrite to recompute)")
            continue
        print(f"Analyzing {path} -> {output_path}")
        _run_single(path, config, output_path, args.overwrite)


def main() -> None:
    args = parse_args()

    config = FeatureConfig(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        enable_stems=args.enable_stems,
        enable_embeddings=args.embeddings,
        disable_essentia=args.disable_essentia,
        disable_madmom=args.disable_madmom,
        max_array_points=args.max_array_points,
    )

    if args.batch_folder:
        _run_batch(args, config)
        return

    _run_single(args.audio, config, args.output, args.overwrite)


if __name__ == "__main__":
    main()
