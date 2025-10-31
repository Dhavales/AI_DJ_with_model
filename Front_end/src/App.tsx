import React, { useState } from 'react';
import { Settings as SettingsIcon, Shuffle, Sparkles } from 'lucide-react';
import { DeckPlayer } from './components/DeckPlayer';
import { MixingControls } from './components/MixingControls';
import { Settings } from './components/Settings';
import { ThemeProvider } from './components/ThemeProvider';
import { Button } from './components/ui/button';
import { TrackList } from './components/TrackList';
import { discoverLibrary } from '@/services/backend';

interface Track {
  id: string;
  title: string;
  artist: string;
  duration: string;
  path?: string;
  cover_url?: string;
}

// Full track pool for shuffling
const allTracks: Track[] = [];

// Get random tracks from pool
const getRandomTracks = (pool: Track[], count: number, exclude: string[] = []): Track[] => {
  const available = pool.filter(track => !exclude.includes(track.id));
  const shuffled = [...available].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
};

function MainApp({ onOpenSettings }: { onOpenSettings: () => void }) {

  // Load library from backend
  const [allTracksState, setAllTracksState] = useState<Track[]>([]);
  React.useEffect(() => {
    const allowed = (import.meta as any)?.env?.VITE_LIBRARY_PATH || '/mnt/nvme/Genie_lib';
    discoverLibrary([allowed], undefined, 200).then(items => {
      // Map backend items into Track (with :mm:ss duration)
      const toMmss = (sec:number)=>{
        const m=Math.floor(sec/60);const s=Math.floor(sec%60);return `${m}:${s.toString().padStart(2,'0')}`;
      };
      const tracks: Track[] = items.map(i=>({ id:i.id, title:i.title, artist:i.artist, duration: toMmss(i.duration), path: i.path, cover_url: i.cover_url }));
      setAllTracksState(tracks);
      const aInitial = getRandomTracks(tracks, 2);
      const bInitial = getRandomTracks(tracks, 2, aInitial.map(t => t.id));
      setDeckATracks(aInitial);
      setDeckBTracks(bInitial);
      setSelectedA(aInitial[0] ?? null);
      setSelectedB(bInitial[0] ?? null);
    }).catch(err=>console.error('discover failed', err));
  }, []);

  const [selectedA, setSelectedA] = useState<Track | null>(null);
  const [selectedB, setSelectedB] = useState<Track | null>(null);
  const [deckATracks, setDeckATracks] = useState<Track[]>([]);
  const [deckBTracks, setDeckBTracks] = useState<Track[]>([]);

  const handleShuffleTracks = () => {
    const newDeckATracks = getRandomTracks(allTracksState, 2);
    const newDeckBTracks = getRandomTracks(allTracksState, 2, newDeckATracks.map(t => t.id));
    setDeckATracks(newDeckATracks);
    setDeckBTracks(newDeckBTracks);
  };

  const handleDeckATrackDelete = (trackId: string) => {
    const remaining = deckATracks.filter(track => track.id !== trackId);
    // Get a new track to replace the deleted one
    const usedIds = [...remaining.map(t => t.id), ...deckBTracks.map(t => t.id)];
    const newTrack = getRandomTracks(allTracksState, 1, usedIds);
    setDeckATracks([...remaining, ...newTrack]);
  };

  const handleDeckBTrackDelete = (trackId: string) => {
    const remaining = deckBTracks.filter(track => track.id !== trackId);
    // Get a new track to replace the deleted one
    const usedIds = [...remaining.map(t => t.id), ...deckATracks.map(t => t.id)];
    const newTrack = getRandomTracks(allTracksState, 1, usedIds);
    setDeckBTracks([...remaining, ...newTrack]);
  };

  const handleDeckATrackReorder = (fromIndex: number, toIndex: number) => {
    const newTracks = [...deckATracks];
    const [movedTrack] = newTracks.splice(fromIndex, 1);
    newTracks.splice(toIndex, 0, movedTrack);
    setDeckATracks(newTracks);
  };

  const handleDeckBTrackReorder = (fromIndex: number, toIndex: number) => {
    const newTracks = [...deckBTracks];
    const [movedTrack] = newTracks.splice(fromIndex, 1);
    newTracks.splice(toIndex, 0, movedTrack);
    setDeckBTracks(newTracks);
  };

  return (
    <div className="min-h-screen bg-background p-3 sm:p-4 md:p-6">
      <div className="max-w-md sm:max-w-2xl lg:max-w-4xl xl:max-w-5xl mx-auto space-y-4 sm:space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between py-2 sm:py-4">
          <h1 className="flex-1 text-center">DJ Mix Studio</h1>
          <Button
            variant="ghost"
            size="icon"
            onClick={onOpenSettings}
            className="flex-shrink-0"
          >
            <SettingsIcon className="w-5 h-5" />
          </Button>
        </div>

        {/* Deck Layout */}
        <div className="grid grid-cols-2 gap-3 sm:gap-4 md:gap-6">
          {/* Deck A */}
          <DeckPlayer 
            deckId="A"
            trackName={selectedA?.title || "Deck A"}
            albumArt={selectedA?.cover_url}
            currentTrack={selectedA}
            onPick={(t) => {
              setSelectedA(t);
              setDeckATracks((prev) => {
                const filtered = prev.filter(p => p.id !== t.id);
                return [t, ...filtered].slice(0, Math.max(2, prev.length));
              });
            }}
          />

          {/* Deck B */}
          <DeckPlayer 
            deckId="B"
            trackName={selectedB?.title || "Deck B"}
            albumArt={selectedB?.cover_url}
            currentTrack={selectedB}
            onPick={(t) => {
              setSelectedB(t);
              setDeckBTracks((prev) => {
                const filtered = prev.filter(p => p.id !== t.id);
                return [t, ...filtered].slice(0, Math.max(2, prev.length));
              });
            }}
          />
        </div>

        {/* Track Lists Section with Shuffle Button */}
        <div className="space-y-3">
          {/* Up Next Headers with Shuffle Button */}
          <div className="relative grid grid-cols-2 gap-3 sm:gap-4 md:gap-6 items-center">
            <div className="text-xs sm:text-sm opacity-70 px-2 text-center">Up Next</div>
            <div className="text-xs sm:text-sm opacity-70 px-2 text-center">Up Next</div>
            
            {/* Shuffle Button - Centered between headers */}
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
              <Button
                size="icon"
                variant="ghost"
                onClick={handleShuffleTracks}
                className="h-8 w-8 text-[rgb(0,255,183)] hover:bg-accent/20 transition-all bg-[rgba(0,0,0,0)]"
              >
                <Sparkles className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Track Lists */}
          <div className="grid grid-cols-2 gap-3 sm:gap-4 md:gap-6">
            {deckATracks.length === 0 ? (
              <div className="text-xs opacity-70 px-2 py-1">No tracks found for Deck A</div>
            ) : (
              <TrackList
                tracks={deckATracks}
                onTrackSelect={(track) => setSelectedA(track)}
                onTrackDelete={handleDeckATrackDelete}
                onTrackReorder={handleDeckATrackReorder}
              />
            )}
            {deckBTracks.length === 0 ? (
              <div className="text-xs opacity-70 px-2 py-1">No tracks found for Deck B</div>
            ) : (
              <TrackList
                tracks={deckBTracks}
                onTrackSelect={(track) => setSelectedB(track)}
                onTrackDelete={handleDeckBTrackDelete}
                onTrackReorder={handleDeckBTrackReorder}
              />
            )}
          </div>
        </div>

        {/* Mixing Controls */}
        <MixingControls deckA={selectedA ? { title: selectedA.title, duration: selectedA.duration, path: selectedA.path } : null} deckB={selectedB ? { title: selectedB.title, duration: selectedB.duration, path: selectedB.path } : null} />
      </div>
    </div>
  );
}

export default function App() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <ThemeProvider>
      {showSettings ? (
        <Settings onClose={() => setShowSettings(false)} />
      ) : (
        <MainApp onOpenSettings={() => setShowSettings(true)} />
      )}
    </ThemeProvider>
  );
}
