import React, { useState } from 'react';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Sparkles } from 'lucide-react';
import { generateMixByPath } from '@/services/backend';

type MixingControlsProps = {
  deckA?: { title?: string; duration?: string; path?: string } | null;
  deckB?: { title?: string; duration?: string; path?: string } | null;
};

export function MixingControls({ deckA, deckB }: MixingControlsProps) {
  const [transitionStyle, setTransitionStyle] = useState('linear');
  const [crossfadeDuration, setCrossfadeDuration] = useState([3]);
  const [crossfadeStyle, setCrossfadeStyle] = useState('beat-align');
  const [mixUrl, setMixUrl] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<{ a: number; cross: number; b: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateMix = () => {
    setError(null);
    setLoading(true);
    if (!deckA?.path || !deckB?.path) {
      setError('Select tracks for both decks');
      setLoading(false);
      return;
    }
    generateMixByPath(deckA.path, deckB.path, {
      crossfade: crossfadeDuration[0],
      a_start: 0,
      b_entry: 0,
      limiter: true,
    })
      .then(({ mixUrl, timeline }) => {
        setMixUrl(mixUrl);
        setTimeline(timeline);
        console.log('Mix ready', { transitionStyle, crossfadeSeconds: crossfadeDuration[0], crossfadeStyle });
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="flex flex-col gap-3 sm:gap-4 p-3 sm:p-4 md:p-6 bg-card border-2 border-border rounded-lg shadow-[0_0_30px_rgba(139,0,255,0.4)]">
      <h3 className="text-center">Mixing Controls</h3>

      {/* Transition Style */}
      <div className="space-y-2">
        <Label htmlFor="transition-style">Transition Style</Label>
        <Select value={transitionStyle} onValueChange={setTransitionStyle}>
          <SelectTrigger id="transition-style">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="linear">Linear</SelectItem>
            <SelectItem value="exponential">Exponential</SelectItem>
            <SelectItem value="echo-out">Echo Out</SelectItem>
            <SelectItem value="cut-echo">Cut + Echo</SelectItem>
            <SelectItem value="filter-sweep">Filter Sweep</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Crossfade Duration */}
      <div className="space-y-2">
        <Label htmlFor="crossfade-duration">Crossfade Duration: {crossfadeDuration[0]}s</Label>
        <Slider
          id="crossfade-duration"
          value={crossfadeDuration}
          onValueChange={setCrossfadeDuration}
          min={1}
          max={30}
          step={0.5}
          className="w-full"
        />
      </div>

      {/* Crossfade Style */}
      <div className="space-y-2">
        <Label htmlFor="crossfade-style">Crossfade Style</Label>
        <Select value={crossfadeStyle} onValueChange={setCrossfadeStyle}>
          <SelectTrigger id="crossfade-style">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="beat-align">Beat Align</SelectItem>
            <SelectItem value="key-lock">Key Lock</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Generate Mix Button */}
      <Button
        onClick={handleGenerateMix}
        className="w-full h-11 sm:h-12 bg-primary hover:bg-primary/90 text-primary-foreground shadow-[0_0_20px_rgba(255,0,255,0.6)]"
        disabled={loading}
      >
        <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
        {loading ? 'Rendering…' : 'Generate Mix'}
      </Button>
      {error && <div className="text-sm text-destructive mt-2">{error}</div>}
      {mixUrl && (
        <div className="mt-3 space-y-2">
          <audio controls src={mixUrl} className="w-full" />
          {timeline && (
            <div className="text-xs opacity-70">
              Timeline — A:{timeline.a.toFixed(1)}s, cross:{timeline.cross.toFixed(1)}s, B:{timeline.b.toFixed(1)}s
            </div>
          )}
        </div>
      )}
    </div>
  );
}
