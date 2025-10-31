import { useState } from 'react';
import { ArrowLeft, User, Palette, Info } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Slider } from './ui/slider';
import { useTheme } from './ThemeProvider';
import { Separator } from './ui/separator';

interface SettingsProps {
  onClose: () => void;
}

export function Settings({ onClose }: SettingsProps) {
  const { hue, setHue } = useTheme();
  const [profile, setProfile] = useState({
    username: 'DJ_Master',
    email: 'dj@mixstudio.com',
    bio: 'Professional DJ and music producer'
  });

  return (
    <div className="min-h-screen bg-background p-3 sm:p-4 md:p-6">
      <div className="max-w-md sm:max-w-2xl lg:max-w-4xl xl:max-w-5xl mx-auto space-y-4 sm:space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4 py-2 sm:py-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="flex-shrink-0"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <h1>Settings</h1>
        </div>

        {/* Settings Content */}
        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="profile">
              <User className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">Profile</span>
            </TabsTrigger>
            <TabsTrigger value="appearance">
              <Palette className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">Appearance</span>
            </TabsTrigger>
            <TabsTrigger value="about">
              <Info className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">About</span>
            </TabsTrigger>
          </TabsList>

          {/* Profile Tab */}
          <TabsContent value="profile" className="space-y-4 mt-4">
            <div className="bg-card border-2 border-border rounded-lg p-4 sm:p-6 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  value={profile.username}
                  onChange={(e) => setProfile({ ...profile, username: e.target.value })}
                  className="bg-input-background"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={profile.email}
                  onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                  className="bg-input-background"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="bio">Bio</Label>
                <Input
                  id="bio"
                  value={profile.bio}
                  onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
                  className="bg-input-background"
                />
              </div>

              <Separator className="my-4" />

              <Button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground">
                Save Profile
              </Button>
            </div>
          </TabsContent>

          {/* Appearance Tab */}
          <TabsContent value="appearance" className="space-y-4 mt-4">
            <div className="bg-card border-2 border-border rounded-lg p-4 sm:p-6 space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="hue-slider">
                    Color Scheme Hue: {hue}°
                  </Label>
                  <p className="text-xs sm:text-sm opacity-70">
                    Adjust the hue to change the app's color scheme
                  </p>
                </div>

                <Slider
                  id="hue-slider"
                  value={[hue]}
                  onValueChange={(value) => setHue(value[0])}
                  min={0}
                  max={360}
                  step={1}
                  className="w-full"
                />

                {/* Color Preview */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
                  <div className="space-y-2">
                    <div 
                      className="h-16 rounded-lg border-2 border-border"
                      style={{ backgroundColor: `hsl(${hue}, 100%, 50%)` }}
                    />
                    <p className="text-xs text-center opacity-70">Primary</p>
                  </div>
                  <div className="space-y-2">
                    <div 
                      className="h-16 rounded-lg border-2 border-border"
                      style={{ backgroundColor: `hsl(${(hue + 180) % 360}, 100%, 50%)` }}
                    />
                    <p className="text-xs text-center opacity-70">Accent</p>
                  </div>
                  <div className="space-y-2">
                    <div 
                      className="h-16 rounded-lg border-2 border-border"
                      style={{ backgroundColor: `hsl(${(hue + 30) % 360}, 100%, 50%)` }}
                    />
                    <p className="text-xs text-center opacity-70">Secondary</p>
                  </div>
                  <div className="space-y-2">
                    <div 
                      className="h-16 rounded-lg border-2 border-border"
                      style={{ backgroundColor: `hsl(${hue}, 60%, 12%)` }}
                    />
                    <p className="text-xs text-center opacity-70">Card</p>
                  </div>
                </div>
              </div>

              <Separator className="my-4" />

              {/* Preset Themes */}
              <div className="space-y-3">
                <Label>Preset Themes</Label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setHue(300)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(300,100%,50%)]" />
                    <span className="text-xs">Magenta</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(0)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(0,100%,50%)]" />
                    <span className="text-xs">Red</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(240)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(240,100%,50%)]" />
                    <span className="text-xs">Blue</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(120)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(120,100%,50%)]" />
                    <span className="text-xs">Green</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(180)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(180,100%,50%)]" />
                    <span className="text-xs">Cyan</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(60)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(60,100%,50%)]" />
                    <span className="text-xs">Yellow</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(30)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(30,100%,50%)]" />
                    <span className="text-xs">Orange</span>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setHue(270)}
                    className="h-auto flex-col gap-2 p-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-[hsl(270,100%,50%)]" />
                    <span className="text-xs">Purple</span>
                  </Button>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* About Tab */}
          <TabsContent value="about" className="space-y-4 mt-4">
            <div className="bg-card border-2 border-border rounded-lg p-4 sm:p-6 space-y-4">
              <div className="text-center space-y-2">
                <h2>DJ Mix Studio</h2>
                <p className="opacity-70">Version 1.0.0</p>
              </div>

              <Separator />

              <div className="space-y-2 text-sm opacity-70">
                <p>
                  A professional DJ mixing application with dual decks, 
                  advanced mixing controls, and customizable appearance.
                </p>
                <p className="mt-4">
                  Features:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>Dual deck playback</li>
                  <li>Advanced transition controls</li>
                  <li>Track search and queue</li>
                  <li>Customizable color themes</li>
                  <li>Responsive design</li>
                </ul>
              </div>

              <Separator />

              <div className="text-center text-xs opacity-60">
                © 2025 DJ Mix Studio. All rights reserved.
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
