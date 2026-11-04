# Demo video source

The README demo is rendered from code with [Remotion](https://www.remotion.dev/) (React + CSS).
It is a **simulated playback**: token timings are generated from the rates set in `src/Root.tsx`,
not captured from a live run. Change `slowRate` / `fastRate` / `footnote` there to match your own
measurements.

```bash
npm install
npx remotion studio src/index.ts                                   # live preview
npx remotion render src/index.ts DuelWide ../../assets/demo-wide.mp4
npx remotion render src/index.ts DuelTall ../../assets/demo-linkedin-4x5.mp4
npx remotion still  src/index.ts Social   ../../assets/social-preview.png
npx remotion still  src/index.ts Logo     ../../assets/logo.png
```

GIF for the README:

```bash
ffmpeg -i ../../assets/demo-wide.mp4 -vf "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" ../../assets/demo.gif
```

Remotion is free for individuals and small companies; larger companies need a
[company license](https://www.remotion.dev/license).
