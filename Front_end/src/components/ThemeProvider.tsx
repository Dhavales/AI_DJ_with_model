import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface ThemeContextType {
  hue: number;
  setHue: (hue: number) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [hue, setHue] = useState(() => {
    const saved = localStorage.getItem('theme-hue');
    return saved ? parseInt(saved) : 300; // Default to magenta (300deg)
  });

  useEffect(() => {
    localStorage.setItem('theme-hue', hue.toString());
    updateThemeColors(hue);
  }, [hue]);

  return (
    <ThemeContext.Provider value={{ hue, setHue }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

function updateThemeColors(hue: number) {
  const root = document.documentElement;
  
  // Calculate complementary color (180deg opposite)
  const complementaryHue = (hue + 180) % 360;
  
  // Primary color (main hue)
  const primary = `hsl(${hue}, 100%, 50%)`;
  const primaryDark = `hsl(${hue}, 100%, 20%)`;
  const primaryLight = `hsl(${hue}, 100%, 70%)`;
  
  // Accent color (complementary)
  const accent = `hsl(${complementaryHue}, 100%, 50%)`;
  
  // Secondary (slightly rotated from primary)
  const secondary = `hsl(${(hue + 30) % 360}, 100%, 50%)`;
  
  // Background and card colors with hue tint
  const background = `hsl(${hue}, 70%, 5%)`;
  const card = `hsl(${hue}, 60%, 12%)`;
  const muted = `hsl(${hue}, 50%, 20%)`;
  
  root.style.setProperty('--background', background);
  root.style.setProperty('--foreground', primary);
  root.style.setProperty('--card', card);
  root.style.setProperty('--card-foreground', accent);
  root.style.setProperty('--popover', card);
  root.style.setProperty('--popover-foreground', accent);
  root.style.setProperty('--primary', primary);
  root.style.setProperty('--primary-foreground', background);
  root.style.setProperty('--secondary', secondary);
  root.style.setProperty('--secondary-foreground', accent);
  root.style.setProperty('--muted', muted);
  root.style.setProperty('--muted-foreground', primaryLight);
  root.style.setProperty('--accent', accent);
  root.style.setProperty('--accent-foreground', background);
  root.style.setProperty('--border', `hsla(${hue}, 100%, 50%, 0.3)`);
  root.style.setProperty('--input', `hsla(${hue}, 100%, 50%, 0.2)`);
  root.style.setProperty('--input-background', `hsla(${hue}, 100%, 50%, 0.1)`);
  root.style.setProperty('--switch-background', secondary);
  root.style.setProperty('--ring', `hsla(${complementaryHue}, 100%, 50%, 0.5)`);
}
