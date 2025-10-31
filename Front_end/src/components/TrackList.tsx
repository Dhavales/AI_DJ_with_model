import { Music2, X, GripVertical } from 'lucide-react';
import { Button } from './ui/button';

interface Track {
  id: string;
  title: string;
  artist: string;
  duration: string;
}

interface TrackListProps {
  tracks: Track[];
  onTrackSelect?: (track: Track) => void;
  onTrackDelete?: (trackId: string) => void;
  onTrackReorder?: (fromIndex: number, toIndex: number) => void;
}

export function TrackList({ tracks, onTrackSelect, onTrackDelete, onTrackReorder }: TrackListProps) {
  const handleDragStart = (e: React.DragEvent, index: number) => {
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', index.toString());
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, toIndex: number) => {
    e.preventDefault();
    const fromIndex = parseInt(e.dataTransfer.getData('text/plain'));
    if (fromIndex !== toIndex && onTrackReorder) {
      onTrackReorder(fromIndex, toIndex);
    }
  };

  return (
    <div className="space-y-1">
      {tracks.map((track, index) => (
          <div
            key={track.id}
            draggable
            onDragStart={(e) => handleDragStart(e, index)}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, index)}
            className="w-full p-2 sm:p-3 bg-card/50 hover:bg-card border border-border/50 rounded-lg transition-all hover:shadow-[0_0_10px_rgba(255,0,255,0.2)] cursor-move group"
          >
            <div className="flex items-start gap-2">
              <GripVertical className="w-3 h-3 sm:w-4 sm:h-4 mt-0.5 flex-shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
              <button
                onClick={() => onTrackSelect?.(track)}
                className="flex items-start gap-2 flex-1 min-w-0 text-left"
              >
                <Music2 className="w-3 h-3 sm:w-4 sm:h-4 mt-0.5 flex-shrink-0 text-primary" />
                <div className="flex-1 min-w-0">
                  <div className="truncate text-xs sm:text-sm">{track.title}</div>
                  <div className="text-xs opacity-60 truncate">{track.artist}</div>
                </div>
                <div className="text-xs opacity-60 flex-shrink-0">{track.duration}</div>
              </button>
              {onTrackDelete && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={() => onTrackDelete(track.id)}
                >
                  <X className="w-3 h-3 text-destructive" />
                </Button>
              )}
            </div>
          </div>
        ))}
    </div>
  );
}
