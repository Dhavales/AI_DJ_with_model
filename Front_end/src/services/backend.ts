
export type LibraryItem = {
  id: string;
  title: string;
  artist: string;
  duration: number;
  bpm: number;
  key: string;
  cover_url: string;
  path: string;
};

export type Timeline = { a: number; cross: number; b: number };

function apiBase(): string {
  const fromEnv = (import.meta as any)?.env?.VITE_API_BASE as string | undefined;
  return (fromEnv && fromEnv.trim()) || 'http://localhost:8001';
}

export async function discoverLibrary(allowed: string[], q?: string, limit?: number): Promise<LibraryItem[]> {
  const params = new URLSearchParams();
  for (const a of allowed) params.append('allowed', a);
  if (q && q.trim()) params.set('q', q.trim());
  if (typeof limit === 'number') params.set('limit', String(limit));
  const url = `${apiBase()}/library/discover?${params.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`discover failed: ${res.status}`);
  return (await res.json()) as LibraryItem[];
}

export async function generateMixByPath(
  deckAPath: string,
  deckBPath: string,
  settings: { crossfade: number; a_start?: number; b_entry?: number; limiter?: boolean }
): Promise<{ mixUrl: string; timeline: Timeline; format: string }>{
  const url = `${apiBase()}/mix/generate_by_path`;
  const body = {
    deckA: { path: deckAPath, start: settings.a_start ?? 0 },
    deckB: { path: deckBPath, start: settings.b_entry ?? 0 },
    settings,
  };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`mix_by_path failed: ${res.status} ${text}`);
  }
  const json = await res.json();
  const b64: string = json.mix_b64;
  const format: string = json.format || 'audio/wav';
  return { mixUrl: `data:${format};base64,${b64}`, timeline: json.timeline, format };
}

export async function fetchAnalysisByPath(path: string): Promise<any> {
  const url = `${apiBase()}/analysis/by_path?` + new URLSearchParams({ path }).toString();
  const res = await fetch(url);
  if (!res.ok) throw new Error('analysis fetch failed: ' + res.status);
  return await res.json();
}
