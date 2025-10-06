"""Streamlit interface that emulates a Pioneer DDJ-FLX4 style workflow.

The app reads analysis JSON files produced by ``feauture_extraction.py`` and a
training bundle from ``train_recommender.py``. A JSON instruction file provides
the deck configuration and preferred mixing parameters. The UI presents two
decks with album art, transport controls, a mixing style selector, and a global
seek bar at the bottom. When FFmpeg is available the backend renders mixed
audio with several transition styles.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - streamlit is optional during tests
    raise SystemExit("Streamlit is required for the DDJ-FLX4 interface") from exc

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - optional mixing backend
    AudioSegment = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from pydub import AudioSegment as AudioSegmentType
else:
    AudioSegmentType = Any

try:  # pragma: no cover - optional lightweight mixing backend
    import librosa  # type: ignore[assignment]
except Exception:  # pylint: disable=broad-except
    librosa = None  # type: ignore[assignment]

try:  # pragma: no cover - optional lightweight mixing backend
    import soundfile as sf  # type: ignore[assignment]
except Exception:  # pylint: disable=broad-except
    sf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional album art extraction
    import importlib

    MutagenFile = getattr(importlib.import_module("mutagen"), "File")
except Exception:  # pylint: disable=broad-except
    MutagenFile = None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    track_id: int
    json_path: Path
    audio_path: Path
    display_name: str
    artist: Optional[str]
    tempo_bpm: float
    cluster: Optional[int] = None


@dataclass
class InstructionSet:
    deck_a_query: str
    deck_b_query: str
    mix_style: str
    crossfade_ms: int
    effects: List[str]
    auto_recommend: bool
    seek_position: float


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _parse_key(key_str: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    if not key_str:
        return None, None
    try:
        root, mode = key_str.split()
    except ValueError:
        parts = key_str.split(" ")
        root = parts[0]
        mode = parts[1] if len(parts) > 1 else "major"

    pitch_classes = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
        "Cb": 11,
    }
    root_idx = pitch_classes.get(root.capitalize())
    if root_idx is None:
        return None, mode.lower()
    return root_idx, mode.lower()


def _safe_get(data: Dict[str, Any], *keys: str, default: float = np.nan) -> float:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if isinstance(current, (int, float)) and not math.isnan(float(current)):
        return float(current)
    return default


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="ignore")
    return json.loads(text)


def _cache_filename(report_root: Path, allowed_roots: Optional[List[Path]]) -> Path:
    base_path = report_root if report_root.is_dir() else report_root.parent
    cache_key_parts = [str(report_root)]
    if allowed_roots:
        cache_key_parts.extend(sorted(str(root) for root in allowed_roots))
    digest = hashlib.md5("|".join(cache_key_parts).encode("utf-8")).hexdigest()
    return base_path / f".analysis_cache_{digest}.pkl"


def load_analysis_dataset(report_root: Path, allowed_roots: Optional[List[Path]] = None) -> DataFrame:
    cache_path = _cache_filename(report_root, allowed_roots)
    if cache_path.exists():
        try:
            cached_df = pd.read_pickle(cache_path)
            if isinstance(cached_df, DataFrame) and not cached_df.empty:
                cached_df = cached_df.reset_index(drop=True)
                cached_df.index.name = "track_id"
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cached_df.to_pickle(cache_path)
                except Exception:  # pragma: no cover - best effort cache repair
                    pass
                return cached_df
        except Exception:  # pragma: no cover - cache corruption fallback
            try:
                cache_path.unlink()
            except Exception:
                pass

    json_files = sorted(report_root.glob("**/*.json"))
    rows: List[Dict[str, Any]] = []
    for json_path in json_files:
        if json_path.name.startswith("._"):
            continue
        try:
            payload = _read_json(json_path)
        except json.JSONDecodeError:
            continue
        analysis = payload.get("analysis", {})
        metadata = payload.get("metadata", {})
        audio_path_str = metadata.get("path") or str(json_path)
        audio_path = Path(audio_path_str)

        if allowed_roots:
            permitted = False
            for root in allowed_roots:
                root_path = Path(root)
                try:
                    if root_path == audio_path.parent or root_path in audio_path.parents:
                        permitted = True
                        break
                except Exception:  # pragma: no cover - defensive
                    continue
            if not permitted:
                continue
        key_root, key_mode = _parse_key(analysis.get("tonal", {}).get("key"))

        track_index = len(rows)
        rows.append(
            {
                "track_id": track_index,
                "path": str(json_path),
                "audio_path": str(audio_path),
                "display_name": audio_path.stem,
                "artist": metadata.get("artist"),
                "tempo_bpm": _safe_get(analysis, "rhythm", "tempo_bpm_global"),
                "danceability": _safe_get(analysis, "rhythm", "danceability_score"),
                "groove_conf": _safe_get(analysis, "rhythm", "groove_confidence"),
                "swing_ratio": _safe_get(analysis, "rhythm", "swing_analysis", "swing_ratio"),
                "key_root": float(key_root) if key_root is not None else np.nan,
                "key_mode_minor": 1.0 if key_mode == "minor" else 0.0 if key_mode else np.nan,
                "spectral_centroid": _safe_get(analysis, "tonal", "spectral_centroid", "mean"),
                "spectral_flatness": _safe_get(analysis, "tonal", "spectral_flatness", "mean"),
                "dissonance": _safe_get(analysis, "tonal", "dissonance_index"),
                "rms_mean": _safe_get(analysis, "dynamics", "rms", "mean"),
                "crest_factor": _safe_get(analysis, "dynamics", "crest_factor"),
                "lufs": _safe_get(analysis, "dynamics", "lufs"),
                "valence": _safe_get(analysis, "mood", "valence"),
                "arousal": _safe_get(analysis, "mood", "arousal"),
                "energy": _safe_get(analysis, "mood", "energy"),
                "bass_presence": _safe_get(analysis, "instrumentation", "bass_presence"),
                "mid_presence": _safe_get(analysis, "instrumentation", "mid_presence"),
                "treble_presence": _safe_get(analysis, "instrumentation", "treble_presence"),
                "percussive_ratio": _safe_get(analysis, "instrumentation", "percussive_ratio"),
            }
        )

    if not rows:
        raise RuntimeError(f"No analysis JSON files found under {report_root}")

    df = pd.DataFrame(rows)
    df.set_index("track_id", inplace=True)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    except Exception:  # pragma: no cover - cache write best effort
        pass
    return df


def _prepare_feature_matrix(df: DataFrame, feature_columns: List[str], median_values: Series, scaler) -> np.ndarray:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    median_values = median_values.reindex(feature_columns).fillna(0.0)
    frame = frame.fillna(median_values).fillna(0.0)
    scaled = scaler.transform(frame)
    return np.nan_to_num(scaled)


def _track_from_row(row: Series) -> TrackRecord:
    cluster_val = row.get("cluster")
    cluster = int(cluster_val) if pd.notna(cluster_val) else None
    return TrackRecord(
        track_id=int(row.name),
        json_path=Path(row["path"]),
        audio_path=Path(row["audio_path"]),
        display_name=str(row["display_name"]),
        artist=row.get("artist"),
        tempo_bpm=float(row.get("tempo_bpm", 0.0) or 0.0),
        cluster=cluster,
    )


def _resolve_track(df: DataFrame, query: str) -> TrackRecord:
    matches = df[df["display_name"].str.contains(query, case=False, na=False)]
    if matches.empty:
        matches = df[df["audio_path"].str.contains(query, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Could not find a track matching '{query}'")
    if len(matches) > 1:
        preview = ", ".join(matches["display_name"].head(5))
        st.sidebar.info(f"Multiple matches for '{query}', using '{matches['display_name'].iloc[0]}'\nCandidates: {preview}")
    row = matches.iloc[0]
    return _track_from_row(row)


def _load_album_art(audio_path: Path) -> Optional[str]:
    if MutagenFile is None or not audio_path.exists():
        return None
    try:
        muta = MutagenFile(audio_path)
    except Exception:  # pragma: no cover - mutagen can fail on some files
        return None
    if not muta:
        return None
    art_data: Optional[bytes] = None
    if hasattr(muta, "tags") and muta.tags:
        pictures = []
        for tag in muta.tags.values():
            if isinstance(tag, bytes):
                pictures.append(tag)
            elif hasattr(tag, "data"):
                pictures.append(tag.data)
        if pictures:
            art_data = pictures[0]
    if art_data is None and hasattr(muta, "pictures"):
        pics = getattr(muta, "pictures")
        if pics:
            art_data = pics[0].data
    if not art_data:
        return None
    encoded = base64.b64encode(art_data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------


def _recommend_tracks(
    bundle: Dict[str, Any],
    train_meta: DataFrame,
    query_vector: np.ndarray,
    cluster_id: Optional[int],
    exclude_paths: Iterable[str],
    top_n: int,
    require_cluster: bool = True,
) -> List[Tuple[Series, float]]:
    nn_model = bundle["nearest_neighbors"]
    neighbors = min(top_n * 4, len(train_meta)) or 1
    distances, indices = nn_model.kneighbors(query_vector.reshape(1, -1), n_neighbors=neighbors)

    exclude_set = {str(path) for path in exclude_paths}
    recommendations: List[Tuple[Series, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        candidate = train_meta.iloc[int(idx)]
        if str(candidate["path"]) in exclude_set:
            continue
        if require_cluster and cluster_id is not None and candidate.get("cluster") != cluster_id:
            continue
        recommendations.append((candidate, float(dist)))
        if len(recommendations) >= top_n:
            break
    return recommendations


# ---------------------------------------------------------------------------
# Mixing back-end
# ---------------------------------------------------------------------------


def _match_tempo(segment: AudioSegmentType, target_bpm: float, source_bpm: float) -> AudioSegmentType:
    if AudioSegment is None:
        raise RuntimeError("pydub not available")
    if target_bpm <= 0 or source_bpm <= 0:
        return segment
    ratio = target_bpm / source_bpm
    if ratio <= 0:
        return segment
    new_frame_rate = int(segment.frame_rate * ratio)
    stretched = segment._spawn(segment.raw_data, overrides={"frame_rate": new_frame_rate})
    return stretched.set_frame_rate(segment.frame_rate)


def _apply_effects(segment: AudioSegmentType, effects: List[str]) -> AudioSegmentType:
    if AudioSegment is None:
        return segment
    output = segment
    if "echo_tail" in effects:
        echo = segment[-4000:].fade_out(3500) - 12
        output = output.append(echo, crossfade=0)
    if "filter_drop" in effects:
        output = output.low_pass_filter(1600)
    return output


def _compute_rms_envelope(y: np.ndarray, sr: int, window_sec: float, hop_factor: float = 4.0) -> Tuple[np.ndarray, np.ndarray, int]:
    frame_length = max(256, int(window_sec * sr))
    frame_length = min(frame_length, len(y))
    if frame_length <= 1:
        return np.array([float(np.sqrt(np.mean(y ** 2)))]) if len(y) else np.array([0.0]), np.array([0]), 1
    hop_length = max(1, int(frame_length / hop_factor))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    positions = np.arange(len(rms)) * hop_length
    return rms, positions, frame_length


def _find_quiet_transition_point(y: np.ndarray, sr: int, crossfade_samples: int) -> int:
    if len(y) <= crossfade_samples:
        return max(0, len(y) - crossfade_samples)
    search_back_sec = max(6.0, crossfade_samples / sr * 2.0)
    window_sec = min(2.5, max(1.0, search_back_sec / 4.0))
    rms, positions, frame_length = _compute_rms_envelope(y, sr, window_sec)
    search_start = max(0, len(y) - int(search_back_sec * sr))
    mask = positions >= search_start
    if not np.any(mask):
        return max(0, len(y) - crossfade_samples)
    sub_positions = positions[mask]
    sub_rms = rms[mask]
    idx = int(np.argmin(sub_rms))
    candidate = int(sub_positions[idx] - frame_length // 2)
    candidate = max(0, min(candidate, max(0, len(y) - crossfade_samples)))
    return candidate


def _find_hook_position(y: np.ndarray, sr: int, crossfade_samples: int) -> int:
    if len(y) <= crossfade_samples:
        return 0
    search_front_sec = min(90.0, max(30.0, len(y) / sr * 0.6))
    window_sec = 2.0
    rms, positions, frame_length = _compute_rms_envelope(y, sr, window_sec, hop_factor=3.0)
    search_end = int(search_front_sec * sr)
    mask = positions <= search_end
    if not np.any(mask):
        return 0
    sub_positions = positions[mask]
    sub_rms = rms[mask]
    idx = int(np.argmax(sub_rms))
    hook = int(sub_positions[idx] - frame_length // 3)
    hook = max(0, min(hook, max(0, len(y) - crossfade_samples)))
    return hook


def render_mix(
    track_a: TrackRecord,
    track_b: TrackRecord,
    tempo_a: float,
    tempo_b: float,
    style: str,
    crossfade_ms: int,
    effects: List[str],
) -> Optional[Tuple[bytes, str]]:
    if AudioSegment is None:
        return _render_mix_numpy(track_a, track_b, tempo_a, tempo_b, style, crossfade_ms)
    if not track_a.audio_path.exists() or not track_b.audio_path.exists():
        return _render_mix_numpy(track_a, track_b, tempo_a, tempo_b, style, crossfade_ms)
    try:
        seg_a = AudioSegment.from_file(track_a.audio_path)
        seg_b = AudioSegment.from_file(track_b.audio_path)
    except FileNotFoundError:
        return _render_mix_numpy(track_a, track_b, tempo_a, tempo_b, style, crossfade_ms)
    except Exception:  # pragma: no cover - fallback when ffmpeg decoding fails
        return _render_mix_numpy(track_a, track_b, tempo_a, tempo_b, style, crossfade_ms)

    seg_a = _apply_effects(seg_a, effects)
    seg_b = _apply_effects(seg_b, effects)

    if style == "beatmatch_crossfade":
        seg_b = _match_tempo(seg_b, tempo_a, tempo_b)
        mixed = seg_a.append(seg_b, crossfade=crossfade_ms)
    elif style == "echo_transition":
        seg_b = _match_tempo(seg_b, tempo_a, tempo_b)
        tail = seg_a[-crossfade_ms:].fade_out(crossfade_ms)
        intro = seg_b[:crossfade_ms].fade_in(crossfade_ms)
        mixed = seg_a[:-crossfade_ms] + tail.overlay(intro)
    elif style == "cut_mix":
        seg_b = _match_tempo(seg_b, tempo_a, tempo_b)
        mixed = seg_a[: max(0, len(seg_a) - crossfade_ms)] + seg_b
    elif style == "loop_roll":
        loop_len = min(4000, len(seg_a) // 4)
        loop = seg_a[-loop_len:].fade_in(250).fade_out(250)
        looped = loop * 4
        seg_b = _match_tempo(seg_b, tempo_a, tempo_b)
        mixed = seg_a[:-loop_len] + looped.overlay(seg_b[: len(looped)]) + seg_b[len(looped) :]
    else:
        seg_b = _match_tempo(seg_b, tempo_a, tempo_b)
        mixed = seg_a.append(seg_b, crossfade=crossfade_ms)

    buffer = io.BytesIO()
    try:
        mixed.export(buffer, format="mp3")
    except Exception:  # pragma: no cover - fallback when mp3 export not available
        return _render_mix_numpy(track_a, track_b, tempo_a, tempo_b, style, crossfade_ms)
    return buffer.getvalue(), "audio/mp3"


def _render_mix_numpy(
    track_a: TrackRecord,
    track_b: TrackRecord,
    tempo_a: float,
    tempo_b: float,
    style: str,
    crossfade_ms: int,
) -> Optional[Tuple[bytes, str]]:
    if librosa is None or sf is None:
        return None
    try:
        y_a, sr_a = librosa.load(str(track_a.audio_path), sr=None, mono=True)
        y_b, sr_b = librosa.load(str(track_b.audio_path), sr=None, mono=True)
    except Exception:  # pragma: no cover - defensive
        return None

    sr = sr_a
    if sr_b != sr_a:
        y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)

    if tempo_b > 0 and tempo_a > 0:
        rate = tempo_b / tempo_a
        if rate > 0:
            try:
                y_b = librosa.effects.time_stretch(y_b, rate=rate)
            except Exception:  # pragma: no cover - defensive
                pass

    crossfade_samples = max(1, int(sr * crossfade_ms / 1000))
    crossfade_samples = min(crossfade_samples, len(y_a), len(y_b))

    transition_start = _find_quiet_transition_point(y_a, sr, crossfade_samples)
    hook_start = _find_hook_position(y_b, sr, crossfade_samples)

    if style == "cut_mix":
        head = y_a[:transition_start]
        tail_b = y_b[hook_start:]
        mix = np.concatenate([head, tail_b])
    else:
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)

        head = y_a[:transition_start]
        tail_a = y_a[transition_start : transition_start + crossfade_samples]
        if len(tail_a) < crossfade_samples:
            padding = np.zeros(crossfade_samples - len(tail_a), dtype=tail_a.dtype)
            tail_a = np.concatenate([tail_a, padding])

        hook_segment = y_b[hook_start : hook_start + crossfade_samples]
        if len(hook_segment) < crossfade_samples:
            padding = np.zeros(crossfade_samples - len(hook_segment), dtype=hook_segment.dtype)
            hook_segment = np.concatenate([hook_segment, padding])

        cross = tail_a * fade_out + hook_segment * fade_in

        if style == "loop_roll":
            loop_len = min(len(tail_a), crossfade_samples)
            loop = tail_a[-loop_len:]
            loop = np.tile(loop, 2)
            loop_env = np.linspace(1.0, 0.2, len(loop), dtype=loop.dtype)
            loop = loop * loop_env
            cross = np.pad(cross, (0, max(0, len(loop) - len(cross))), mode="constant")
            cross = cross[: len(loop)] + loop[: len(cross)]

        post_b = y_b[hook_start + crossfade_samples :]
        mix = np.concatenate([head, cross, post_b])

    buffer = io.BytesIO()
    mix = np.nan_to_num(mix, nan=0.0, posinf=1.0, neginf=-1.0)
    sf.write(buffer, mix.astype(np.float32), sr, format="WAV")
    return buffer.getvalue(), "audio/wav"


# ---------------------------------------------------------------------------
# Instruction handling
# ---------------------------------------------------------------------------


def load_instructions(path: Path) -> InstructionSet:
    if not path.exists():
        return InstructionSet(
            deck_a_query="",
            deck_b_query="",
            mix_style="beatmatch_crossfade",
            crossfade_ms=8000,
            effects=[],
            auto_recommend=True,
            seek_position=0.0,
        )
    payload = _read_json(path)
    return InstructionSet(
        deck_a_query=payload.get("deckA", {}).get("query", ""),
        deck_b_query=payload.get("deckB", {}).get("query", ""),
        mix_style=payload.get("mix_style", "beatmatch_crossfade"),
        crossfade_ms=int(payload.get("crossfade_ms", 8000)),
        effects=list(payload.get("effects", [])),
        auto_recommend=bool(payload.get("auto_recommend", True)),
        seek_position=float(payload.get("seek_position", 0.0)),
    )


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


def _load_cached_bundle(model_path: Path):
    @st.cache_resource(show_spinner=False)
    def _loader(path_str: str):
        return joblib.load(path_str)

    return _loader(str(model_path))


def _load_cached_dataset(report_path: Path, allowed_roots: Optional[List[Path]]):
    @st.cache_resource(show_spinner=False)
    def _loader(path_str: str, allowed: Tuple[str, ...]):
        roots = [Path(item) for item in allowed] if allowed else None
        return load_analysis_dataset(Path(path_str), roots)

    allowed_tuple: Tuple[str, ...] = tuple(str(root) for root in allowed_roots) if allowed_roots else tuple()
    return _loader(str(report_path), allowed_tuple)


def _get_feature_matrix(bundle: Dict[str, Any], df: DataFrame):
    @st.cache_resource(show_spinner=False)
    def _loader(feature_columns: Tuple[str, ...], median_values: Tuple[float, ...], serialized_df: str):
        columns = list(feature_columns)
        med_series = Series(list(median_values), index=columns)
        scaler = bundle["scaler"]
        frame = pd.read_json(serialized_df)
        return _prepare_feature_matrix(frame, columns, med_series, scaler)

    feature_columns = tuple(bundle["feature_columns"])
    median_values = bundle.get("median_impute")
    if median_values is None:
        median_values = Series(0.0, index=bundle["feature_columns"])
    elif not isinstance(median_values, Series):
        median_values = Series(median_values, index=bundle["feature_columns"])
    else:
        median_values = median_values.reindex(bundle["feature_columns"]).fillna(0.0)

    return _loader(feature_columns, tuple(median_values.values), df[bundle["feature_columns"]].to_json())


def build_ui(args: argparse.Namespace) -> None:
    st.set_page_config(page_title="DDJ-FLX4 Virtual Deck", layout="wide")
    st.title("🎛️ DDJ-FLX4 Virtual Deck")

    instructions = load_instructions(args.instructions)

    bundle = _load_cached_bundle(args.model)
    allowed_roots = args.allowed_folder or None
    df = _load_cached_dataset(args.reports, allowed_roots)

    if df.empty:
        st.error(
            "No analysis reports matched the allowed folders. Ensure you've generated JSON reports for the selected library and the `--allowed-folder` arguments point to the correct directories."
        )
        return

    feature_columns = bundle["feature_columns"]
    median_values = bundle.get("median_impute")
    if median_values is None:
        median_values = Series(0.0, index=feature_columns)
    elif not isinstance(median_values, Series):
        median_values = Series(median_values, index=feature_columns)
    else:
        median_values = median_values.reindex(feature_columns).fillna(0.0)

    scaled_matrix = _prepare_feature_matrix(df, feature_columns, median_values, bundle["scaler"])
    kmeans = bundle["kmeans"]
    cluster_ids = kmeans.predict(scaled_matrix)
    df["cluster"] = cluster_ids

    default_a = instructions.deck_a_query or df["display_name"].iloc[0]
    if instructions.deck_b_query:
        default_b = instructions.deck_b_query
    else:
        remaining = df[df["display_name"] != default_a]
        default_b = remaining["display_name"].iloc[0] if not remaining.empty else df["display_name"].iloc[min(1, len(df) - 1)]

    with st.sidebar:
        st.header("Instructions JSON")
        st.write(f"Loading configuration from `{args.instructions}`")
        st.json(
            {
                "deckA": {"query": instructions.deck_a_query},
                "deckB": {"query": instructions.deck_b_query},
                "mix_style": instructions.mix_style,
                "crossfade_ms": instructions.crossfade_ms,
                "effects": instructions.effects,
                "auto_recommend": instructions.auto_recommend,
            }
        )
        st.caption("Edit the JSON file and refresh to update defaults.")
        st.subheader("Active Library Folders")
        if allowed_roots:
            for root in allowed_roots:
                st.caption(str(root))
        else:
            st.caption("Using all available analysis reports")

    col_left, col_center, col_right = st.columns([3, 2, 3])

    with col_left:
        st.subheader("Deck A")
        deck_a_query = st.text_input("Search Deck A", value=default_a)
    with col_right:
        st.subheader("Deck B")
        deck_b_query = st.text_input("Search Deck B", value=default_b)

    try:
        deck_a = _resolve_track(df, deck_a_query)
        deck_b = _resolve_track(df, deck_b_query)
    except ValueError as exc:
        st.error(str(exc))
        return

    if deck_a.track_id == deck_b.track_id:
        alt_rows = df.drop(deck_a.track_id)
        if alt_rows.empty:
            st.error("Need at least two distinct tracks for mixing. Add more analysis files or adjust the search queries.")
            return
        deck_b = _track_from_row(alt_rows.iloc[0])
        st.info(f"Deck B auto-adjusted to '{deck_b.display_name}' to avoid duplicating Deck A.")

    col_left, col_center, col_right = st.columns([3, 2, 3])

    with col_left:
        art_a = _load_album_art(deck_a.audio_path)
        if art_a:
            st.image(art_a, use_container_width=True)
        else:
            st.image("https://placehold.co/600x600?text=Deck+A", use_container_width=True)
        st.write(f"**{deck_a.display_name}**")
        if deck_a.artist:
            st.caption(deck_a.artist)
        st.metric("Tempo", f"{deck_a.tempo_bpm:.1f} BPM")
        st.metric("Cluster", deck_a.cluster)
        st.caption("Preview via mix player below.")

    with col_right:
        art_b = _load_album_art(deck_b.audio_path)
        if art_b:
            st.image(art_b, use_container_width=True)
        else:
            st.image("https://placehold.co/600x600?text=Deck+B", use_container_width=True)
        st.write(f"**{deck_b.display_name}**")
        if deck_b.artist:
            st.caption(deck_b.artist)
        st.metric("Tempo", f"{deck_b.tempo_bpm:.1f} BPM")
        st.metric("Cluster", deck_b.cluster)
        st.caption("Preview via mix player below.")

    with col_center:
        st.subheader("Mixer")
        mix_style = st.selectbox(
            "Mixing Style",
            options=[
                "beatmatch_crossfade",
                "echo_transition",
                "loop_roll",
                "cut_mix",
            ],
            index=[
                "beatmatch_crossfade",
                "echo_transition",
                "loop_roll",
                "cut_mix",
            ].index(instructions.mix_style)
            if instructions.mix_style in ["beatmatch_crossfade", "echo_transition", "loop_roll", "cut_mix"]
            else 0,
        )
        crossfade_ms = st.slider("Crossfade Length (ms)", 2000, 15000, instructions.crossfade_ms, step=500)
        effects = st.multiselect(
            "Effects",
            options=["echo_tail", "filter_drop"],
            default=[effect for effect in instructions.effects if effect in {"echo_tail", "filter_drop"}],
        )

    progress = st.slider("Mix Progress", 0.0, 1.0, instructions.seek_position, step=0.01, key="global_seek")

    trigger_mix = st.button("Generate Mix")
    mix_state_key = "rendered_mix"
    if mix_state_key not in st.session_state:
        st.session_state[mix_state_key] = None

    if instructions.auto_recommend:
        train_meta = bundle["metadata"].copy().reset_index(drop=True)
        train_meta["cluster"] = bundle["kmeans"].labels_
        train_meta["display_name"] = train_meta["title"].apply(lambda s: Path(str(s)).stem)
        exclude = [str(deck_a.json_path), str(deck_b.json_path)]
        vec_a = scaled_matrix[deck_a.track_id]
        vec_b = scaled_matrix[deck_b.track_id]
        recs_a = _recommend_tracks(bundle, train_meta, vec_a, deck_a.cluster, exclude, 3, require_cluster=True)
        if not recs_a:
            recs_a = _recommend_tracks(bundle, train_meta, vec_a, deck_a.cluster, exclude, 3, require_cluster=False)
        recs_b = _recommend_tracks(bundle, train_meta, vec_b, deck_b.cluster, exclude, 3, require_cluster=True)
        if not recs_b:
            recs_b = _recommend_tracks(bundle, train_meta, vec_b, deck_b.cluster, exclude, 3, require_cluster=False)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Suggestions for Deck A")
            for candidate, distance in recs_a:
                st.write(f"- {candidate['display_name']} (distance {distance:.3f})")
        with col_right:
            st.subheader("Suggestions for Deck B")
            for candidate, distance in recs_b:
                st.write(f"- {candidate['display_name']} (distance {distance:.3f})")

    if trigger_mix:
        with st.spinner("Rendering mix..."):
            mix_output = render_mix(
                deck_a,
                deck_b,
                deck_a.tempo_bpm,
                deck_b.tempo_bpm,
                mix_style,
                crossfade_ms,
                effects,
            )
        if mix_output is None:
            st.session_state[mix_state_key] = None
            st.warning("Mix could not be rendered. Ensure FFmpeg or the lightweight fallback dependencies are installed.")
        else:
            mix_bytes, mime_type = mix_output
            file_ext = "mp3" if mime_type == "audio/mp3" else "wav"
            st.session_state[mix_state_key] = {
                "bytes": mix_bytes,
                "mime": mime_type,
                "file_name": f"mix_{deck_a.display_name}_{deck_b.display_name}.{file_ext}",
                "deck_a": deck_a.display_name,
                "deck_b": deck_b.display_name,
            }

    mix_result = st.session_state.get(mix_state_key)
    if mix_result and (
        mix_result.get("deck_a") != deck_a.display_name or mix_result.get("deck_b") != deck_b.display_name
    ):
        mix_result = None
        st.session_state[mix_state_key] = None
    if mix_result:
        st.success(f"Mix ready: {mix_result['deck_a']} -> {mix_result['deck_b']}")
        st.audio(mix_result["bytes"], format=mix_result["mime"])
        st.download_button(
            label="Download Mix",
            data=mix_result["bytes"],
            file_name=mix_result["file_name"],
            mime=mix_result["mime"],
        )

    st.caption("Progress slider mimics the jog wheel / transport position. Adjust it to rehearse transitions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDJ-FLX4 style virtual DJ interface")
    parser.add_argument("--model", type=Path, default=Path("models/recommender.joblib"), help="Path to the recommender joblib bundle")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Directory with analysis JSON files")
    parser.add_argument("--instructions", type=Path, default=Path("instructions.json"), help="JSON file providing default deck configuration")
    parser.add_argument(
        "--allowed-folder",
        action="append",
        help="Restrict audio library to these folders (relative or absolute). Use multiple times for more than one folder.",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else (base_dir / path).resolve()

    args.model = _resolve(args.model)
    args.reports = _resolve(args.reports)
    args.instructions = _resolve(args.instructions)

    allowed_values = args.allowed_folder or []
    if allowed_values:
        args.allowed_folder = [_resolve(Path(item)) for item in allowed_values]
    else:
        default_roots = [
            (base_dir / "Music_Data" / "songs_david_guetta").resolve(),
            (base_dir / "Music_Data" / "songs_pop").resolve(),
        ]
        args.allowed_folder = [folder for folder in default_roots if folder.exists()]
        if not args.allowed_folder:
            args.allowed_folder = []

    return args


def main() -> None:
    args = parse_args()
    build_ui(args)


if __name__ == "__main__":
    main()
