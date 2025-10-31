"""
FastAPI backend exposing the core non-UI functions from dj_interface.py.

Endpoints:
- GET  /health                        -> ok
- GET  /library/discover?allowed=...  -> discover_library() results
- POST /audio/synthesize              -> synthesize_tone() to generate a test tone WAV
- POST /mix/generate                  -> generate_mix() to return a WAV mix and a timeline

This module reuses the mixing approach from dj_interface.py (pydub if available,
otherwise numpy/librosa crossfade), but removes Streamlit/UI concerns so it can
serve the Front_end.
"""
from __future__ import annotations

import base64
import io
import wave
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from tinytag import TinyTag  # type: ignore
except Exception:  # pragma: no cover
    TinyTag = None  # type: ignore

try:
    from pydub import AudioSegment  # type: ignore
except Exception:  # pragma: no cover
    AudioSegment = None  # type: ignore

try:
    from mutagen import File as MutagenFile  # type: ignore
except Exception:  # pragma: no cover
    MutagenFile = None  # type: ignore


# ----------------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------------

@dataclass
class Track:
    id: str
    title: str
    artist: str
    bpm: int
    key: str
    duration: float
    cover_url: str


class DeckIn(BaseModel):
    # Minimal fields needed to render a mix
    title: Optional[str] = None
    artist: Optional[str] = None
    format: Optional[str] = Field(default="wav", description="File format hint for decoding, e.g. mp3, wav, m4a")
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    audio_b64: str = Field(..., description="Base64-encoded bytes of a mono WAV or other supported format")


class MixSettings(BaseModel):
    crossfade: float = 8.0
    a_start: float = 0.0
    b_entry: float = 0.0
    limiter: bool = True
    style: Optional[str] = None
    beat_align: Optional[bool] = None
    key_lock: Optional[bool] = None


class MixResponse(BaseModel):
    mix_b64: str
    timeline: Dict[str, float]
    format: str = "audio/wav"


class DiscoverResponseItem(BaseModel):
    id: str
    title: str
    artist: str
    duration: float
    bpm: int
    key: str
    cover_url: str
    path: str


# ----------------------------------------------------------------------------
# Library discovery (ported from Streamlit UI without visuals)
# ----------------------------------------------------------------------------

def discover_library(allowed: Optional[List[str]]) -> List[Track]:
    if not allowed:
        # basic mock when no folder is provided
        artists = ["DJ Nova", "Midnight Echo", "Solar Atlas", "Pulse Theory"]
        tracks: List[Track] = []
        for idx in range(12):
            bpm = int(100 + (idx * 3) % 32)
            duration = 180 + (idx * 10) % 180
            tracks.append(
                Track(
                    id=f"mock-{idx}",
                    title=f"Night Voyage {idx + 1}",
                    artist=artists[idx % len(artists)],
                    bpm=bpm,
                    key="?",
                    duration=float(duration),
                    cover_url=f"https://picsum.photos/seed/dj{idx+1}/320/320",
                )
            )
        return tracks

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
            from urllib.parse import quote
            cover = f"/audio/cover_by_path?path={quote(str(audio_path.resolve()))}"
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
    return tracks


# ----------------------------------------------------------------------------
# Cover extraction helpers
# ----------------------------------------------------------------------------


def _extract_cover_bytes(path: Path) -> Optional[Tuple[bytes, str]]:
    if MutagenFile is None:
        return None
    try:
        mf = MutagenFile(str(path))
    except Exception:
        return None
    if not mf:
        return None
    # FLAC pictures
    try:
        if hasattr(mf, 'pictures') and mf.pictures:
            pic = mf.pictures[0]
            data = getattr(pic, 'data', None)
            mime = getattr(pic, 'mime', 'image/jpeg') or 'image/jpeg'
            if data:
                return data, mime
    except Exception:
        pass
    # MP3 ID3 APIC
    try:
        if hasattr(mf, 'tags') and mf.tags and hasattr(mf.tags, 'getall'):
            apics = mf.tags.getall('APIC')
            if apics:
                ap = apics[0]
                data = getattr(ap, 'data', None)
                mime = getattr(ap, 'mime', 'image/jpeg') or 'image/jpeg'
                if data:
                    return data, mime
    except Exception:
        pass
    # MP4/M4A covr
    try:
        if hasattr(mf, 'tags') and mf.tags and 'covr' in mf.tags:
            covr = mf.tags['covr']
            if isinstance(covr, (list, tuple)) and covr:
                data = bytes(covr[0])
                return data, 'image/jpeg'
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Audio helpers (ported)
# ----------------------------------------------------------------------------

def synthesize_tone(duration: float, freq: float = 220.0, sample_rate: int = 44100) -> Tuple[bytes, int]:
    duration = max(duration, 1.0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.25 * np.sin(2 * np.pi * freq * t)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue(), sample_rate


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


def audiosegment_from_bytes(audio_bytes: bytes, fmt: str) -> Optional[AudioSegment]:
    if AudioSegment is None:
        return None
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        return seg.set_channels(1)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Mixing (ported)
# ----------------------------------------------------------------------------

def generate_mix(deck_a: Dict[str, Any], deck_b: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[bytes, Dict[str, float]]:
    crossfade = float(settings.get("crossfade", 8.0))
    a_start = float(settings.get("a_start", 0.0))
    b_entry = float(settings.get("b_entry", 0.0))

    # Preferred: pydub for simplicity/limiter
    if AudioSegment is not None:
        fmt_a = str(deck_a.get("format", "wav"))
        fmt_b = str(deck_b.get("format", "wav"))
        seg_a = audiosegment_from_bytes(deck_a["audio_bytes"], fmt_a)
        seg_b = audiosegment_from_bytes(deck_b["audio_bytes"], fmt_b)
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

    # Fallback: numpy crossfade. Expect WAV input already.
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


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------

app = FastAPI(title="AI DJ Backend", version="0.1.0")

# Enable CORS for local dev (Vite runs on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/library/discover", response_model=List[DiscoverResponseItem])
async def api_discover(
    allowed: Optional[List[str]] = Query(default=None),
    q: Optional[str] = Query(default=None, description="Case-insensitive substring to match in title/artist/path"),
    limit: Optional[int] = Query(default=None, ge=1, le=5000),
):
    """Discover tracks from allowed folders; optionally filter by a search query and limit results."""
    all_tracks = discover_library(allowed)
    qnorm = (q or "").strip().lower()
    if qnorm:
        def _match(t: Track) -> bool:
            return (
                qnorm in (t.title or "").lower()
                or qnorm in (t.artist or "").lower()
                or qnorm in Path(t.id).name.lower()
            )
        filtered = [t for t in all_tracks if _match(t)]
    else:
        filtered = all_tracks

    if limit is not None:
        filtered = filtered[: int(limit)]

    results: List[DiscoverResponseItem] = []
    for t in filtered:
        results.append(
            DiscoverResponseItem(
                id=t.id,
                title=t.title,
                artist=t.artist,
                duration=float(t.duration),
                bpm=int(t.bpm),
                key=t.key,
                cover_url=t.cover_url,
                path=t.id,
            )
        )
    return results


@app.post("/audio/synthesize", response_model=MixResponse)
async def api_synthesize(duration: float = 8.0, freq: float = 220.0):
    wav_bytes, _ = synthesize_tone(duration, freq)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return MixResponse(mix_b64=b64, timeline={"a": duration, "cross": 0.0, "b": 0.0})


class MixGenerateRequest(BaseModel):
    deckA: DeckIn
    deckB: DeckIn
    settings: Optional[MixSettings] = None


@app.post("/mix/generate", response_model=MixResponse)
async def api_generate(req: MixGenerateRequest):
    try:
        # Decode incoming audio
        a_bytes = base64.b64decode(req.deckA.audio_b64)
        b_bytes = base64.b64decode(req.deckB.audio_b64)
        deck_a = {"format": (req.deckA.format or "wav"), "audio_bytes": a_bytes}
        deck_b = {"format": (req.deckB.format or "wav"), "audio_bytes": b_bytes}
        settings = (req.settings or MixSettings()).model_dump()
        wav_bytes, timeline = generate_mix(deck_a, deck_b, settings)
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return MixResponse(mix_b64=b64, timeline=timeline)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"mix_failed: {exc}")



from fastapi import Response

class DeckPathIn(BaseModel):
    path: str
    start: float = 0.0

class MixByPathRequest(BaseModel):
    deckA: DeckPathIn
    deckB: DeckPathIn
    settings: Optional[MixSettings] = None


def _mix_by_path(path_a: str, path_b: str, settings: Dict[str, Any], start_a: float, start_b: float) -> Tuple[bytes, Dict[str, float]]:
    # Prefer pydub for arbitrary formats when available
    if AudioSegment is not None:
        try:
            seg_a = AudioSegment.from_file(path_a).set_channels(1)
            seg_b = AudioSegment.from_file(path_b).set_channels(1)
            seg_a_clip = seg_a[start_a * 1000:]
            seg_b_clip = seg_b[start_b * 1000:]
            crossfade_ms = max(0, int(float(settings.get('crossfade', 8.0)) * 1000))
            mixed = seg_a_clip.append(seg_b_clip, crossfade=crossfade_ms)
            if settings.get('limiter', True):
                mixed = mixed.normalize()
            buf = io.BytesIO()
            mixed.export(buf, format='wav')
            timeline = {
                'a': len(seg_a_clip) / 1000.0,
                'cross': min(float(settings.get('crossfade', 8.0)), len(seg_a_clip) / 1000.0, len(seg_b_clip) / 1000.0),
                'b': len(seg_b_clip) / 1000.0,
            }
            return buf.getvalue(), timeline
        except Exception:
            pass
    # Fallback: WAV-only via wave module
    def _load_wav(path: str) -> Tuple[np.ndarray, int]:
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            return samples, sr
    if path_a.lower().endswith('.wav') and path_b.lower().endswith('.wav'):
        a_data, sr_a = _load_wav(path_a)
        b_data, sr_b = _load_wav(path_b)
        sr = min(sr_a, sr_b)
        a_clip = a_data[int(start_a * sr):]
        b_clip = b_data[int(start_b * sr):]
        crossfade = float(settings.get('crossfade', 8.0))
        crossfade_samples = int(min(crossfade, len(a_clip) / sr, len(b_clip) / sr) * sr)
        if crossfade_samples > 0:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            blended = a_clip[:crossfade_samples] * fade_out + b_clip[:crossfade_samples] * fade_in
            mixed = np.concatenate([a_clip[:-crossfade_samples], blended, b_clip[crossfade_samples:]])
        else:
            mixed = np.concatenate([a_clip, b_clip])
        buf = io.BytesIO()
        write_wav_bytes(buf, mixed, sr)
        timeline = {'a': len(a_clip) / sr, 'cross': crossfade, 'b': len(b_clip) / sr}
        return buf.getvalue(), timeline
    raise ValueError('Unsupported format without pydub/ffmpeg; supply WAVs or enable pydub+ffmpeg')


@app.post('/mix/generate_by_path', response_model=MixResponse)
async def api_generate_by_path(req: MixByPathRequest):
    try:
        a = Path(req.deckA.path)
        b = Path(req.deckB.path)
        if not a.exists() or not b.exists():
            raise HTTPException(status_code=404, detail='file_not_found')
        settings = (req.settings or MixSettings()).model_dump()
        wav_bytes, timeline = _mix_by_path(str(a), str(b), settings, req.deckA.start, req.deckB.start)
        b64 = base64.b64encode(wav_bytes).decode('ascii')
        return MixResponse(mix_b64=b64, timeline=timeline)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'mix_failed: {exc}')

# Optional: quick stream-by-path for future player usage
@app.get('/audio/stream_by_path')
async def api_stream_by_path(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail='file_not_found')
    # naive MIME guess
    ext = p.suffix.lower()
    mime = 'audio/wav' if ext == '.wav' else 'audio/mpeg' if ext in ('.mp3', '.mpeg') else 'application/octet-stream'
    data = p.read_bytes()
    return Response(content=data, media_type=mime)

# Fallback 1x1 PNG (transparent)
FALLBACK_COVER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAOkNydsAAAAASUVORK5CYII="
)


@app.get('/audio/cover_by_path')
async def api_cover_by_path(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail='file_not_found')
    data = _extract_cover_bytes(p)
    if data is None:
        # Return a small transparent PNG as a fallback
        return Response(content=base64.b64decode(FALLBACK_COVER_PNG_B64), media_type='image/png')
    content, mime = data
    return Response(content=content, media_type=mime or 'image/jpeg')


@app.get('/analysis/by_path')
async def api_analysis_by_path(path: str):
    pth = Path(path)
    if not pth.exists() or not pth.is_file():
        raise HTTPException(status_code=404, detail='file_not_found')
    js = pth.with_suffix('.json')
    if not js.exists():
        import os, glob
        reports_root = os.environ.get('ANALYSIS_REPORTS_DIR', 'reports')
        try:
            pattern = os.path.join(reports_root, '**', pth.stem + '.json')
            matches = glob.glob(pattern, recursive=True)
            if matches:
                js = Path(matches[0])
        except Exception:
            pass
    if not js.exists():
        raise HTTPException(status_code=404, detail='analysis_not_found')
    try:
        text = js.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = js.read_text(encoding='latin-1', errors='ignore')
    import json as _json
    return JSONResponse(content=_json.loads(text))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
