import { useEffect, useRef, useState, useMemo } from 'react';
import { Play, Pause, SkipBack, SkipForward, Search } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Slider } from './ui/slider';

import { ImageWithFallback } from './figma/ImageWithFallback';
import { discoverLibrary, LibraryItem, fetchAnalysisByPath } from '@/services/backend';

interface DeckPlayerProps {
  deckId: 'A' | 'B';
  albumArt?: string;
  trackName?: string;
  currentTrack?: { path?: string; title?: string; duration?: string } | null;
  onPick?: (track: { id: string; title: string; artist: string; duration: string; path: string; cover_url?: string }) => void;
}

export function DeckPlayer({ deckId, albumArt, trackName, currentTrack, onPick }: DeckPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [durationSec, setDurationSec] = useState(0);
  const [currentSec, setCurrentSec] = useState(0);
  const [results, setResults] = useState<LibraryItem[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const [seekPosition, setSeekPosition] = useState([0]);
  const [searchQuery, setSearchQuery] = useState('');
  const [analysis, setAnalysis] = useState<any | null>(null);

  useEffect(() => {
    const q = searchQuery.trim();
    if (q.length < 2) { setResults([]); return; }
    const allowed = (import.meta as any)?.env?.VITE_LIBRARY_PATH || '/mnt/nvme/Genie_lib';
    let cancelled = false;
    discoverLibrary([allowed], q, 6)
      .then(items => { if (!cancelled) setResults(items); })
      .catch(() => { if (!cancelled) setResults([]); });
    return () => { cancelled = true; };
  }, [searchQuery]);
  useEffect(() => {
    if (!currentTrack?.path) { setAnalysis(null); return; }
    fetchAnalysisByPath(currentTrack.path).then(setAnalysis).catch(() => setAnalysis(null));
  }, [currentTrack?.path]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    setIsPlaying(false);
    setCurrentSec(0);
    if (currentTrack?.path) {
      const base = (import.meta as any)?.env?.VITE_API_BASE || 'http://localhost:8001';
      const url = base + '/audio/stream_by_path?path=' + encodeURIComponent(currentTrack.path);
      el.src = url;
      el.load();
    } else {
      el.removeAttribute('src');
    }
  }, [currentTrack?.path]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onLoaded = () => setDurationSec(el.duration || 0);
    const onTime = () => setCurrentSec(el.currentTime || 0);
    const onEnded = () => setIsPlaying(false);
    el.addEventListener('loadedmetadata', onLoaded);
    el.addEventListener('timeupdate', onTime);
    el.addEventListener('ended', onEnded);
    return () => {
      el.removeEventListener('loadedmetadata', onLoaded);
      el.removeEventListener('timeupdate', onTime);
      el.removeEventListener('ended', onEnded);
    };
  }, []);

  const togglePlay = () => {
    const el = audioRef.current; if (!el) return;
    if (isPlaying) { el.pause(); setIsPlaying(false); }
    else { el.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false)); }
  };

  const skipBy = (deltaSec: number) => {
    const el = audioRef.current; if (!el) return;
    const dur = durationSec || el.duration || 0;
    const next = Math.max(0, Math.min(dur, (el.currentTime || 0) + deltaSec));
    el.currentTime = next;
  };

  const featureBlocks = useMemo(() => {
  const a: any = analysis || {};
  const an = a.analysis || {};
  const rhythm = an.rhythm || {};
  const tonal = an.tonal || {};
  const mood = an.mood || {};
  const instr = an.instrumentation || {};
  const clamp01 = (x:number)=> Math.max(0, Math.min(1, Number(x)||0));
  const norm = (x:number, lo:number, hi:number)=> clamp01((Number(x) - lo) / (hi - lo));
  const tempo = Number(rhythm.tempo_bpm_global || 0);
  const tempoN = norm(tempo, 60, 180);
  const dance = clamp01(rhythm.danceability_score);
  const energy = clamp01(mood.energy);
  const valence = clamp01(mood.valence);
  const perc = clamp01(instr.percussive_ratio);
  const bass = clamp01(instr.bass_presence);
  const treble = clamp01(instr.treble_presence);
  const key = String(tonal.key || "?");
  return [
  { label: `BPM ${tempo || "—"}`, v: tempoN },
  { label: `Key ${key}`, v: 0.75 },
  { label: "Dance", v: dance },
  { label: "Energy", v: energy },
  { label: "Valence", v: valence },
  { label: "Perc", v: perc },
  { label: "Bass", v: bass },
  { label: "Treble", v: treble },
  ];
  }, [analysis]);

  const percent = durationSec > 0 ? Math.max(0, Math.min(100, (currentSec / durationSec) * 100)) : 0;


  return (
    <div className="flex flex-col gap-2 sm:gap-3 flex-1">
      {/* Search Bar */}
      <div className="relative z-50">
        <Search className="absolute left-2 sm:left-3 top-1/2 -translate-y-1/2 w-3 h-3 sm:w-4 sm:h-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder={`Deck ${deckId}...`}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-7 sm:pl-9 h-8 sm:h-9 bg-card border-border"
        />
        {results.length > 0 && (
          <div className="absolute z-50 mt-1 w-full bg-card border border-border rounded-md shadow-md max-h-56 overflow-auto">
            {results.map((r) => (
              <button
                key={r.id}
                onClick={() => {
                  const toMmss = (sec:number)=>{ const m=Math.floor(sec/60); const s=Math.floor(sec%60); return m + ':' + s.toString().padStart(2,'0'); };
                  onPick?.({ id: r.id, title: r.title, artist: r.artist, duration: toMmss(r.duration), path: r.path, cover_url: r.cover_url });
                  setSearchQuery('');
                  setResults([]);
                }}
                className="w-full text-left px-3 py-2 hover:bg-accent/20 border-b border-border last:border-b-0"
              >
                <div className="text-xs sm:text-sm truncate">{r.title}</div>
                <div className="text-[10px] sm:text-xs opacity-70 truncate">{r.artist}</div>
              </button>
            ))}
          </div>
        )}
      </div>

      <audio ref={audioRef} preload="metadata" style={{ display: "none" }} />

      {/* Deck Label */}
      <div className="text-center opacity-70">Deck {deckId}</div>

      {/* Album Art with Controls */}
      <div
        className="relative aspect-square bg-card border-2 border-border rounded-lg overflow-hidden shadow-[0_0_20px_rgba(255,0,255,0.3)]"
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
        onTouchStart={() => setShowControls(true)}
      >
        {/* Feature blocks overlay */}
        {analysis && featureBlocks && (
          <div className="absolute z-30 left-3 right-3 top-3 grid grid-cols-4 gap-3 pointer-events-none">
            {featureBlocks.map((f, i) => (
              <div key={i} className="h-4 rounded bg-orange-500/30 border border-orange-500/60 shadow-sm" title={`${f.label}: ${Math.round(f.v * 100)}%`}>
                <div className="h-full rounded bg-orange-500" style={{ width: `${Math.round(f.v * 100)}%` }} />
              </div>
            ))}
          </div>
        )}
        {/* Album Art */}
        <ImageWithFallback
          src={albumArt || 'https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=400&h=400&fit=crop'}
          alt={`Deck ${deckId} Album Art`}
          className="w-full h-full object-cover"
        />

        {/* Controls Overlay */}
        <div 
          className={`absolute z-40 inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center gap-2 sm:gap-3 md:gap-4 transition-opacity duration-300 ${
            showControls ? 'opacity-100' : 'opacity-0'
          }`}
        >
          <Button
            size="icon"
            variant="ghost"
            className="w-10 h-10 sm:w-12 sm:h-12 md:w-14 md:h-14 rounded-full bg-primary/20 hover:bg-primary/30 text-primary border border-primary/50"
            onClick={() => skipBy(-5)}
          >
            <SkipBack className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6" />
          </Button>
          
          <Button
            size="icon"
            variant="ghost"
            className="w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 rounded-full bg-accent hover:bg-accent/80 text-accent-foreground shadow-[0_0_20px_rgba(0,255,255,0.6)]"
            onClick={togglePlay}
          >
            {isPlaying ? (
              <Pause className="w-6 h-6 sm:w-8 sm:h-8 md:w-10 md:h-10" fill="currentColor" />
            ) : (
              <Play className="w-6 h-6 sm:w-8 sm:h-8 md:w-10 md:h-10" fill="currentColor" />
            )}
          </Button>
          
          <Button
            size="icon"
            variant="ghost"
            className="w-10 h-10 sm:w-12 sm:h-12 md:w-14 md:h-14 rounded-full bg-primary/20 hover:bg-primary/30 text-primary border border-primary/50"
            onClick={() => skipBy(5)}
          >
            <SkipForward className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6" />
          </Button>
        </div>
      </div>

      {/* Track Info */}
      {trackName && (
        <div className="text-center truncate px-2">
          {trackName}
        </div>
      )}
      {analysis && (
        <div className="px-2 text-xs sm:text-sm space-y-1">
          <div className="flex gap-2"><span className="opacity-70">BPM:</span><span>{analysis.analysis?.rhythm?.tempo_bpm_global ?? '—'}</span></div>
          <div className="flex gap-2"><span className="opacity-70">Key:</span><span>{analysis.analysis?.tonal?.key ?? '—'}</span></div>
          <div className="flex items-center gap-2"><span className="opacity-70">Danceability</span><div className="flex-1 h-1 bg-border rounded"><div style={{width: `${Math.round(((analysis.analysis?.rhythm?.danceability_score ?? 0)*100))}%`}} className="h-1 bg-accent rounded" /></div></div>
          <div className="flex items-center gap-2"><span className="opacity-70">Energy</span><div className="flex-1 h-1 bg-border rounded"><div style={{width: `${Math.round(((analysis.analysis?.mood?.energy ?? 0)*100))}%`}} className="h-1 bg-primary rounded" /></div></div>
        </div>
      )}


      {/* Seek Bar */}
      <div className="space-y-1">
        <Slider
          value={[percent]}
          onValueChange={(v)=>{ const el = audioRef.current; if (!el || !durationSec) return; const p = Array.isArray(v)? v[0]:(v as any); el.currentTime = (Math.max(0, Math.min(100, Number(p))) / 100) * durationSec; }}
          max={100}
          step={0.1}
          className="w-full"
        />
        <div className="flex justify-between text-muted-foreground">
          {(() => { const m=Math.floor(currentSec/60), ss=Math.floor(currentSec%60); return (<span className="text-xs sm:text-sm">{m}:{ss.toString().padStart(2,'0')}</span>);})()}
          {(() => { const dm=Math.floor(durationSec/60), ds=Math.floor(durationSec%60); return (<span className="text-xs sm:text-sm">{dm}:{ds.toString().padStart(2,'0')}</span>);})()}
        </div>
      </div>
    </div>
  );
}
