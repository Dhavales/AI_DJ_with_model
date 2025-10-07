"""Streamlit DJ interface with two decks and central mix controls."""

from __future__ import annotations

import base64
import io
import random
import wave
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

try:  # optional metadata helper
    from tinytag import TinyTag  # type: ignore
except Exception:  # pragma: no cover
    TinyTag = None  # type: ignore

try:  # optional cover extraction
    from mutagen import File as MutagenFile  # type: ignore
except Exception:  # pragma: no cover
    MutagenFile = None  # type: ignore

try:  # optional richer mixing backend
    from pydub import AudioSegment
except Exception:  # pragma: no cover
    AudioSegment = None  # type: ignore


# ---------------------------------------------------------------------------
# Demo data & constants
# ---------------------------------------------------------------------------

PLACEHOLDER_COVERS = [
    "https://picsum.photos/seed/deck101/320/320",
    "https://picsum.photos/seed/deck102/320/320",
    "https://picsum.photos/seed/deck103/320/320",
    "https://picsum.photos/seed/deck104/320/320",
]

KEY_OPTIONS = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]


@dataclass
class Track:
    id: str
    title: str
    artist: str
    bpm: int
    key: str
    duration: float
    cover_url: str


def build_mock_library() -> List[Track]:
    artists = ["DJ Nova", "Midnight Echo", "Solar Atlas", "Pulse Theory"]
    library: List[Track] = []
    for idx in range(12):
        bpm = random.randint(90, 132)
        duration = random.randint(180, 360)
        library.append(
            Track(
                id=f"mock-{idx}",
                title=f"Night Voyage {idx + 1}",
                artist=artists[idx % len(artists)],
                bpm=bpm,
                key=random.choice(KEY_OPTIONS),
                duration=float(duration),
                cover_url=f"https://picsum.photos/seed/dj{idx+1}/320/320",
            )
        )
    return library


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--allowed-folder", action="append")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def discover_library(allowed: Optional[List[str]]) -> List[Track]:
    if not allowed:
        return build_mock_library()
    tracks: List[Track] = []
    exts = {".mp3", ".wav", ".m4a", ".flac"}
    for folder in allowed:
        path = Path(folder)
        if not path.exists():
            continue
        for audio_path in path.rglob("*"):
            if not audio_path.is_file() or audio_path.suffix.lower() not in exts:
                continue
            title = audio_path.stem
            artist = audio_path.parent.name
            duration = 240.0
            bpm = 0
            if TinyTag is not None:
                try:
                    tag = TinyTag.get(str(audio_path))
                    if tag.title:
                        title = tag.title
                    if tag.artist:
                        artist = tag.artist
                    if tag.duration:
                        duration = float(tag.duration)
                    if getattr(tag, "bpm", None):
                        try:
                            bpm = int(tag.bpm)  # type: ignore[attr-defined]
                        except Exception:
                            bpm = 0
                except Exception:
                    pass
            cover = f"https://picsum.photos/seed/{audio_path.stem.replace(' ', '')}/320/320"
            tracks.append(
                Track(
                    id=str(audio_path.resolve()),
                    title=title,
                    artist=artist,
                    bpm=bpm,
                    key="?",
                    duration=duration,
                    cover_url=cover,
                )
            )
    return tracks or build_mock_library()


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def apply_base_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top, #0f1a2e 0%, #070912 55%, #010309 100%);
                color: #f5f7ff;
                font-family: "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
            }
            .deck-panel, .mix-panel {
                background: linear-gradient(182deg, rgba(24,32,54,0.96) 0%, rgba(12,16,32,0.9) 100%);
                border-radius: 24px;
                border: 1px solid rgba(255,255,255,0.12);
                box-shadow: 0 32px 68px rgba(0,0,0,0.55);
                padding: 1.4rem 1.6rem 1.2rem;
            }
            .deck-panel h3, .mix-panel h3 {
                margin: 0 0 1rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 1.05rem;
            }
            .deck-panel .stTextInput>div>div>input,
            .mix-panel .stTextInput>div>div>input {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                color: #fdfefe;
            }
            .deck-panel .stTextInput>div>div>input:focus,
            .mix-panel .stTextInput>div>div>input:focus {
                border-color: rgba(90,210,255,0.8);
                box-shadow: 0 0 0 1px rgba(90,210,255,0.35);
            }
            .cover-preview img {
                border-radius: 18px;
                box-shadow: 0 24px 48px rgba(0,0,0,0.55);
            }
            .timeline {
                margin-top: 1rem;
                height: 14px;
                border-radius: 999px;
                background: rgba(255,255,255,0.12);
                position: relative;
                overflow: hidden;
            }
            .timeline span { position: absolute; top: 0; bottom: 0; }
            .timeline .a { left: 0; background: linear-gradient(90deg, rgba(90,210,255,0.75), rgba(90,210,255,0.25)); }
            .timeline .cross { background: linear-gradient(90deg, rgba(255,99,132,0.85), rgba(255,188,71,0.75)); }
            .timeline .b { right: 0; background: linear-gradient(90deg, rgba(120,255,171,0.75), rgba(120,255,171,0.25)); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def synthesize_tone(duration: float, freq: float = 220.0, sample_rate: int = 44100) -> Tuple[bytes, int]:
    duration = max(duration, 12.0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.25 * np.sin(2 * np.pi * freq * t)
    pcm = (waveform * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue(), sample_rate


def extract_cover_art(file_bytes: bytes) -> Optional[str]:
    if MutagenFile is None:
        return None
    try:
        track = MutagenFile(io.BytesIO(file_bytes))
    except Exception:
        return None
    if not track:
        return None
    pictures = []
    if hasattr(track, "pictures") and track.pictures:
        pictures.extend(track.pictures)
    if hasattr(track, "tags") and track.tags:
        for tag in track.tags.values():
            data = getattr(tag, "data", None)
            if isinstance(tag, bytes):
                data = tag
            if data:
                pictures.append(type("P", (), {"data": data}))
    if not pictures:
        return None
    encoded = base64.b64encode(pictures[0].data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def decode_uploaded_file(uploaded) -> Dict[str, object]:
    file_bytes = uploaded.getvalue()
    fmt = Path(uploaded.name).suffix.lower().lstrip(".") or "wav"
    title = Path(uploaded.name).stem
    artist = "Unknown Artist"
    duration = 240.0
    bpm = random.randint(88, 132)
    musical_key = random.choice(KEY_OPTIONS)
    cover = extract_cover_art(file_bytes)

    if TinyTag is not None:
        try:
            tag = TinyTag.get(io.BytesIO(file_bytes))
            title = tag.title or title
            artist = tag.artist or artist
            if tag.duration:
                duration = float(tag.duration)
        except Exception:
            pass

    sample_rate = 44100
    if AudioSegment is not None:
        try:
            seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
            seg = seg.set_channels(1)
            sample_rate = seg.frame_rate
            duration = len(seg) / 1000.0
            buffer = io.BytesIO()
            seg.export(buffer, format="wav")
            file_bytes = buffer.getvalue()
            fmt = "wav"
        except Exception:
            pass

    return {
        "id": f"upload-{uploaded.name}-{random.randint(0, 9999)}",
        "title": title,
        "artist": artist,
        "bpm": bpm,
        "key": musical_key,
        "duration": duration,
        "format": fmt,
        "sample_rate": sample_rate,
        "audio_bytes": file_bytes,
        "cover": cover or random.choice(PLACEHOLDER_COVERS),
    }


def audiosegment_from_deck(deck: Dict[str, object]) -> Optional[AudioSegment]:
    if AudioSegment is None:
        return None
    fmt = str(deck.get("format", "wav"))
    try:
        seg = AudioSegment.from_file(io.BytesIO(deck["audio_bytes"]), format=fmt)
        return seg.set_channels(1)
    except Exception:
        return None


def decode_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        sr = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return samples, sr


def write_wav_bytes(buffer: io.BytesIO, data: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------


def generate_mix(deck_a: Dict[str, object], deck_b: Dict[str, object], settings: Dict[str, object]):
    crossfade = float(settings.get("crossfade", 8.0))
    a_start = float(settings.get("a_start", 0.0))
    b_entry = float(settings.get("b_entry", 0.0))

    if AudioSegment is not None:
        seg_a = audiosegment_from_deck(deck_a)
        seg_b = audiosegment_from_deck(deck_b)
        if seg_a is not None and seg_b is not None:
            seg_a_clip = seg_a[a_start * 1000 :]
            seg_b_clip = seg_b[b_entry * 1000 :]
            crossfade_ms = max(0, int(crossfade * 1000))
            mixed = seg_a_clip.append(seg_b_clip, crossfade=crossfade_ms)
            if settings.get("limiter", True):
                mixed = mixed.normalize()
            buffer = io.BytesIO()
            mixed.export(buffer, format="wav")
            timeline = {
                "a": len(seg_a_clip) / 1000.0,
                "cross": min(crossfade, len(seg_a_clip) / 1000.0, len(seg_b_clip) / 1000.0),
                "b": len(seg_b_clip) / 1000.0,
            }
            return buffer.getvalue(), timeline

    data_a, sr_a = decode_wav_bytes(deck_a["audio_bytes"])
    data_b, sr_b = decode_wav_bytes(deck_b["audio_bytes"])
    sr = min(sr_a, sr_b)
    a_clip = data_a[int(a_start * sr) :]
    b_clip = data_b[int(b_entry * sr) :]
    crossfade_samples = int(min(crossfade, len(a_clip) / sr, len(b_clip) / sr) * sr)
    if crossfade_samples > 0:
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
        blended = a_clip[:crossfade_samples] * fade_out + b_clip[:crossfade_samples] * fade_in
        mixed = np.concatenate([a_clip[:-crossfade_samples], blended, b_clip[crossfade_samples:]])
    else:
        mixed = np.concatenate([a_clip, b_clip])
    buffer = io.BytesIO()
    write_wav_bytes(buffer, mixed, sr)
    timeline = {
        "a": len(a_clip) / sr,
        "cross": crossfade,
        "b": len(b_clip) / sr,
    }
    return buffer.getvalue(), timeline


def render_timeline(timeline: Dict[str, float]) -> None:
    total = max(timeline.get("a", 0.0) + timeline.get("b", 0.0), 0.1)
    cross = min(timeline.get("cross", 0.0), total)
    a_pct = timeline.get("a", 0.0) / total * 100
    cross_pct = cross / total * 100
    html = (
        f"<div class='timeline'>"
        f"<span class='a' style='width:{max(a_pct - cross_pct/2, 0)}%'></span>"
        f"<span class='cross' style='left:{max(a_pct - cross_pct/2, 0)}%; width:{cross_pct}%'></span>"
        f"<span class='b' style='width:{max(100 - (a_pct - cross_pct/2) - cross_pct, 0)}%'></span>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Deck & mix rendering
# ---------------------------------------------------------------------------


def render_deck(deck_key: str, column, heading: str) -> None:
    with column:
        st.markdown("<div class='deck-panel'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{heading}</h3>")

        library = st.session_state.library
        quick_pick = st.selectbox(
            "Quick pick",
            ["None"] + [f"{t.title} — {t.artist}" for t in library],
            key=f"{deck_key}_quickpick",
        )
        if quick_pick != "None":
            idx = [f"{t.title} — {t.artist}" for t in library].index(quick_pick)
            track = library[idx]
            audio_bytes, sample_rate = synthesize_tone(track.duration, freq=200 + idx * 7)
            st.session_state[deck_key] = {
                "id": track.id,
                "title": track.title,
                "artist": track.artist,
                "bpm": track.bpm,
                "key": track.key,
                "duration": track.duration,
                "format": "wav",
                "sample_rate": sample_rate,
                "audio_bytes": audio_bytes,
                "cover": track.cover_url,
            }

        uploaded = st.file_uploader(
            f"Upload for {heading}",
            type=["mp3", "wav", "m4a"],
            key=f"{deck_key}_upload",
        )
        if uploaded is not None:
            st.session_state[deck_key] = decode_uploaded_file(uploaded)
            st.toast(f"Loaded {uploaded.name} on {heading}")

        deck = st.session_state.get(deck_key)
        if deck:
            st.markdown("<div class='cover-preview'>", unsafe_allow_html=True)
            st.image(deck.get("cover") or random.choice(PLACEHOLDER_COVERS), width=180)
            st.markdown("</div>", unsafe_allow_html=True)
            meta_cols = st.columns(3)
            meta_cols[0].metric("BPM", deck.get("bpm", "—"))
            meta_cols[1].metric("Key", deck.get("key", "—"))
            meta_cols[2].metric("Duration", f"{deck.get('duration', 0):.0f}s")
            st.audio(deck["audio_bytes"], format="audio/wav")
        else:
            st.info("Choose a track to begin.")

        st.markdown("</div>", unsafe_allow_html=True)


def render_mix_panel(column) -> None:
    with column:
        st.markdown("<div class='mix-panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Mix Controls</h3>")
        deck_a = st.session_state.get("deckA")
        deck_b = st.session_state.get("deckB")
        settings = st.session_state["mix_settings"]

        settings["crossfade"] = st.slider("Crossfade (sec)", 0.0, 30.0, settings["crossfade"], key="crossfade")

        if deck_a:
            max_a = float(deck_a.get("duration", 0.0))
            settings["a_start"] = st.slider(
                "Deck A start",
                0.0,
                max(max_a - 0.1, 0.0),
                min(settings.get("a_start", 0.0), max(max_a - 0.1, 0.0)),
                key="a_start",
            )
        else:
            st.caption("Load Deck A to adjust start offset.")
            settings["a_start"] = 0.0

        if deck_b:
            max_b = float(deck_b.get("duration", 0.0))
            settings["b_entry"] = st.slider(
                "Deck B start",
                0.0,
                max(max_b - 0.1, 0.0),
                min(settings.get("b_entry", 0.0), max(max_b - 0.1, 0.0)),
                key="b_entry",
            )
        else:
            st.caption("Load Deck B to adjust start offset.")
            settings["b_entry"] = 0.0

        flags = st.columns(2)
        settings["beat_align"] = flags[0].checkbox("Beat align", value=settings["beat_align"], key="beat_align")
        settings["key_lock"] = flags[1].checkbox("Key lock", value=settings["key_lock"], key="key_lock")
        settings["style"] = st.selectbox(
            "Transition style",
            ["Linear", "Exponential", "Echo Out", "Cut + Echo", "Filter Sweep"],
            index=["Linear", "Exponential", "Echo Out", "Cut + Echo", "Filter Sweep"].index(settings["style"])
            if settings["style"] in ["Linear", "Exponential", "Echo Out", "Cut + Echo", "Filter Sweep"]
            else 0,
            key="style",
        )
        settings["limiter"] = st.checkbox("Limiter", value=settings["limiter"], key="limiter")

        st.markdown("---")
        disabled = not (deck_a and deck_b)
        if st.button("Generate Mix", use_container_width=True, disabled=disabled):
            if disabled:
                st.warning("Load both decks first.")
            else:
                with st.spinner("Rendering transition..."):
                    mix_bytes, timeline = generate_mix(deck_a, deck_b, settings)
                st.session_state["mix_result"] = {
                    "bytes": mix_bytes,
                    "timeline": timeline,
                    "format": "audio/wav",
                }
                st.toast("Mix ready")

        mix_result = st.session_state.get("mix_result")
        if mix_result and mix_result.get("bytes"):
            st.audio(mix_result["bytes"], format=mix_result["format"])
            render_timeline(mix_result.get("timeline", {}))
            st.download_button(
                "Download mix",
                mix_result["bytes"],
                file_name="dj_mix.wav",
                mime="audio/wav",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AI DJ Mixer", layout="wide", page_icon="🎚️")
    apply_base_css()
    args = parse_args()
    st.session_state.setdefault("library", discover_library(args.allowed_folder))
    st.session_state.setdefault("deckA", None)
    st.session_state.setdefault("deckB", None)
    st.session_state.setdefault(
        "mix_settings",
        {"crossfade": 8.0, "a_start": 0.0, "b_entry": 0.0, "beat_align": False, "key_lock": False, "style": "Linear", "limiter": True},
    )
    st.session_state.setdefault("mix_result", None)

    col_a, col_mix, col_b = st.columns([4, 3, 4])
    render_deck("deckA", col_a, "Deck A")
    render_mix_panel(col_mix)
    render_deck("deckB", col_b, "Deck B")


if __name__ == "__main__":
    main()
