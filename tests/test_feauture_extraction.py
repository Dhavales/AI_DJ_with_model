import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feauture_extraction as feat


def test_downsample_array_respects_max_points():
    data = np.arange(10)

    result = feat._downsample_array(data, max_points=5)

    assert len(result) == 5
    assert result[0] == 0
    assert result[-1] == 9


def test_downsample_array_handles_empty_input():
    result = feat._downsample_array(np.array([]), max_points=3)

    assert result == []


def test_stats_filters_nan_and_inf():
    values = np.array([1.0, np.nan, np.inf, 3.0])

    stats = feat._stats(values)

    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_spectral_flux_onset_produces_normalised_non_negative_envelope():
    sr = 22050
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 220 * t).astype(float)

    onset_env = feat._spectral_flux_onset(y, hop_length=256)

    assert onset_env.ndim == 1
    assert onset_env.size > 0
    assert np.all(onset_env >= -1e-9)
    assert onset_env.max() <= 1.0 + 1e-9


def test_polyrhythm_candidates_ignores_infinite_tempi(monkeypatch):
    onset_env = np.array([0.0, 1.0, 0.5])

    def fake_tempogram(onset_envelope, sr, hop_length):
        return np.array([[1.0, 1.0, 1.0], [0.7, 0.7, 0.7], [0.3, 0.3, 0.3]])

    def fake_tempo_frequencies(length, sr):
        return np.array([np.inf, 120.0, 60.0])

    monkeypatch.setattr(feat.librosa.feature, "tempogram", fake_tempogram)
    monkeypatch.setattr(feat.librosa, "tempo_frequencies", fake_tempo_frequencies)

    result = feat._polyrhythm_candidates(onset_env, sr=44100, hop_length=512)

    assert result["primary_tempi"] == [120.0, 60.0]
    assert all(np.isfinite(value) for value in result["primary_tempi"])


def test_chord_progression_identifies_major_triad():
    chroma = np.zeros((12, 8))
    chroma[0] = 1.0
    chroma[4] = 0.8
    chroma[7] = 0.9
    result = feat._estimate_chord_progression(chroma, sr=22050, hop_length=512)

    assert result["progression"][0]["chord"] == "C:maj"
    assert result["histogram"]["C:maj"] >= 1


def test_instrument_presence_band_split():
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    bass = 0.8 * np.sin(2 * np.pi * 60 * t)
    treble = 0.2 * np.sin(2 * np.pi * 4000 * t)
    waveform = bass + treble

    result = feat._instrument_presence(waveform, sr)

    assert "bass_presence" in result
    assert "treble_presence" in result
    assert result["bass_presence"] > result["treble_presence"]


def test_analyze_audio_on_sine_wave(tmp_path):
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.25 * np.sin(2 * np.pi * 440 * t)

    def write_wav(path, data, sample_rate):
        import wave

        clipped = np.clip(data, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int_data.tobytes())

    audio_path = tmp_path / "sine.wav"
    write_wav(audio_path, waveform, sr)

    result = feat.analyze_audio(audio_path, feat.FeatureConfig())

    assert result["metadata"]["sample_rate"] == sr
    assert result["metadata"]["duration_sec"] > 0
    assert "analysis" in result
    # Spot check a couple of analysis sections to ensure structure is populated.
    assert "rhythm" in result["analysis"]
    assert "tonal" in result["analysis"]
    assert isinstance(result["analysis"]["rhythm"]["tempo_bpm_global"], float)
    assert "mood" in result["analysis"]
    assert "instrumentation" in result["analysis"]
