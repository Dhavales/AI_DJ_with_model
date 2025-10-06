"""Train a recommendation model using features extracted by feauture_extraction.py.

This script aggregates the JSON analysis files produced under ``reports/`` (or any
other directory), engineers a flat feature matrix, trains normalisation and nearest
neighbour models, and persists everything needed for downstream recommendation or
auto-mixing workflows.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrackFeatures:
    path: Path
    title: str
    artist: Optional[str]
    features: Dict[str, float]


# ---------------------------------------------------------------------------
# Feature extraction helpers
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

    pitch_classes = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
                     "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
                     "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}
    root_idx = pitch_classes.get(root.capitalize())
    if root_idx is None:
        return None, mode.lower()
    return root_idx, mode.lower()


def _safe_get(data: Dict, *keys, default=np.nan) -> float:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if isinstance(current, (int, float)) and not math.isnan(current):
        return float(current)
    return default


def _collect_track_features(json_path: Path) -> Optional[TrackFeatures]:
    try:
        text = json_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = json_path.read_text(encoding="latin-1", errors="ignore")
    payload = json.loads(text)
    analysis = payload.get("analysis", {})

    key_name = analysis.get("tonal", {}).get("key")
    key_root, key_mode = _parse_key(key_name)

    features: Dict[str, float] = {
        "tempo_bpm": _safe_get(analysis, "rhythm", "tempo_bpm_global"),
        "danceability": _safe_get(analysis, "rhythm", "danceability_score"),
        "groove_conf": _safe_get(analysis, "rhythm", "groove_confidence"),
        "swing_ratio": _safe_get(analysis, "rhythm", "swing_analysis", "swing_ratio"),
        "key_root": float(key_root) if key_root is not None else np.nan,
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

    title = payload.get("metadata", {}).get("path", json_path.name)
    artist = payload.get("metadata", {}).get("artist")

    features["key_mode_minor"] = 1.0 if key_mode == "minor" else 0.0 if key_mode else np.nan

    return TrackFeatures(path=json_path, title=title, artist=artist, features=features)


def load_dataset(report_root: Path) -> pd.DataFrame:
    json_files = sorted(report_root.glob("**/*.json"))
    records: List[TrackFeatures] = []
    for json_path in json_files:
        try:
            record = _collect_track_features(json_path)
        except json.JSONDecodeError:
            continue
        if record:
            records.append(record)

    if not records:
        raise RuntimeError(f"No valid JSON files found under {report_root}")

    rows = []
    for idx, item in enumerate(records):
        data = {"track_id": idx, "path": str(item.path), "title": item.title, "artist": item.artist}
        data.update(item.features)
        rows.append(data)

    df = pd.DataFrame(rows)
    df.set_index("track_id", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_neighbors: int,
    random_state: int,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    feature_columns = [col for col in train_df.columns if col not in {"path", "title", "artist"}]
    if not feature_columns:
        raise RuntimeError("No numeric features available for training.")

    train_features = train_df[feature_columns].astype(float)
    median_values = train_features.median()
    train_features = train_features.fillna(median_values)
    train_features = train_features.fillna(0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_train = np.nan_to_num(X_train, copy=False)

    effective_neighbors = min(n_neighbors, len(train_df)) or 1
    nn_model = NearestNeighbors(metric="cosine", n_neighbors=effective_neighbors)
    nn_model.fit(X_train)

    cluster_count = max(2, min(25, max(2, len(train_df) // 10)))
    kmeans = KMeans(n_clusters=cluster_count, random_state=random_state, n_init="auto")
    kmeans.fit(X_train)

    silhouette = None
    if len(train_df) >= cluster_count * 5:
        silhouette = silhouette_score(X_train, kmeans.labels_)

    metrics: Dict[str, float] = {}
    if silhouette is not None:
        metrics["silhouette_score"] = float(silhouette)

    if len(test_df) > 0:
        test_features = test_df[feature_columns].astype(float)
        test_features = test_features.fillna(median_values)
        test_features = test_features.fillna(0.0)
        X_test = scaler.transform(test_features)
        X_test = np.nan_to_num(X_test, copy=False)
        distances, _ = nn_model.kneighbors(X_test)
        metrics["mean_distance_to_train"] = float(np.mean(distances[:, 0]))
        metrics["median_distance_to_train"] = float(np.median(distances[:, 0]))

    bundle = {
        "scaler": scaler,
        "nearest_neighbors": nn_model,
        "kmeans": kmeans,
        "feature_columns": feature_columns,
        "metadata": train_df[["path", "title", "artist"]],
        "median_impute": median_values,
    }

    return bundle, metrics


def save_bundle(bundle: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a song recommendation model from extracted features.")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Directory containing analysis JSON files.")
    parser.add_argument("--model-out", type=Path, default=Path("models/recommender.joblib"), help="Output path for the trained bundle.")
    parser.add_argument("--neighbors", type=int, default=10, help="Number of neighbours to store for recommendations.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of tracks reserved for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting and clustering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.reports}…")
    df = load_dataset(args.reports)
    print(f"Loaded {len(df)} tracks.")

    if len(df) < 2:
        raise RuntimeError("Need at least two tracks to train a recommender.")

    if args.test_size <= 0 or args.test_size >= 1:
        train_df, test_df = df, df.iloc[0:0]
    else:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
        )
        train_df = df.loc[train_idx].sort_index()
        test_df = df.loc[test_idx].sort_index()

    print(f"Training on {len(train_df)} tracks, evaluating on {len(test_df)} tracks.")

    bundle, metrics = train_models(train_df, test_df, n_neighbors=args.neighbors, random_state=args.random_state)

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")

    save_bundle(bundle, args.model_out)
    print(f"Model bundle saved to {args.model_out}")


if __name__ == "__main__":
    main()
