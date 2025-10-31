export type DeckBrief = {
  title?: string;
  bpm?: number;
  duration?: number;
  cue?: number;
  analysis?: {
    bpm?: number;
    barSec?: number;
    duration?: number;
    energyPeaksSec?: number[];
    hookWindowSec?: { start: number; end: number };
    outroStartSec?: number;
  };
};

export type MixStep = {
  atSeconds: number;
  deck: 'A' | 'B' | 'MASTER';
  action:
    | 'PLAY'
    | 'PAUSE'
    | 'CUE'
    | 'JUMP_TO_CUE'
    | 'SET_TEMPO'
    | 'NUDGE'
    | 'EQ'
    | 'FILTER'
    | 'KEYLOCK'
    | 'XFADER'
    | 'LOOP'
    | 'HOTCUE'
    | 'SYNC'
    | 'COMMENT'
    | 'SEEK'
    | 'SET_HOTCUE'
    | 'GAIN'
    | 'ANALYZE';
  params?: Record<string, unknown>;
};

export type MixPlan = {
  version: string;
  plan: { overview: string; steps: MixStep[] };
  finalSettings?: {
    keyLockA?: boolean;
    keyLockB?: boolean;
    tempoA?: number;
    tempoB?: number;
    crossfader?: number;
  };
};

function apiBase(): string {
  const fromEnv = (import.meta as any)?.env?.VITE_API_BASE as string | undefined;
  return (fromEnv && fromEnv.trim()) || 'http://localhost:3001';
}

export async function planMix(deckA: DeckBrief, deckB: DeckBrief): Promise<MixPlan> {
  const url = `${apiBase()}/mix/plan`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ deckA, deckB }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Mix plan request failed: ${res.status} ${text}`);
  }
  return (await res.json()) as MixPlan;
}
