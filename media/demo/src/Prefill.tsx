import React from 'react';
import {AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Bolt} from './Logo';

const ACCENT = '#facc15';
const MONO = '"JetBrains Mono", "Cascadia Code", Consolas, "SF Mono", monospace';

export const PREFILL_SCENE = 5.2; // seconds of video spent on this scene
const BARS_START = 0.8;
const BARS_END = 3.6; // fast engine finishes here (real video time)

const fmtTime = (s: number) => {
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
};

const Lane: React.FC<{
  name: string;
  fast: boolean;
  rate: number;
  total: number;
  simT: number;
  font: number;
  appear: number;
}> = ({name, fast, rate, total, simT, font, appear}) => {
  const done = Math.min(total, rate * simT);
  const finished = done >= total;
  const full = total / rate;
  return (
    <div
      style={{
        padding: `${font * 1.4}px ${font * 1.8}px`,
        borderRadius: 18,
        background: 'linear-gradient(180deg, #0d1220 0%, #080b14 100%)',
        border: `1.5px solid ${fast ? 'rgba(250,204,21,0.55)' : 'rgba(148,163,184,0.25)'}`,
        boxShadow: fast ? '0 0 60px rgba(250,204,21,0.12)' : 'none',
        opacity: appear,
        transform: `translateY(${(1 - appear) * 24}px)`,
      }}
    >
      <div style={{display: 'flex', alignItems: 'baseline', gap: 16}}>
        <div style={{fontSize: font * 1.6, fontWeight: 800, color: fast ? ACCENT : '#cbd5e1'}}>
          {fast ? '⚡ ' : ''}
          {name}
        </div>
        <div style={{marginLeft: 'auto', display: 'flex', alignItems: 'baseline', gap: 8}}>
          <span
            style={{
              fontSize: font * 2.6,
              fontWeight: 800,
              color: fast ? ACCENT : '#e2e8f0',
              fontVariantNumeric: 'tabular-nums',
              textShadow: fast ? '0 0 30px rgba(250,204,21,0.45)' : 'none',
            }}
          >
            {simT > 0 ? rate.toLocaleString('en-US') : '—'}
          </span>
          <span style={{fontSize: font * 1.1, color: '#94a3b8', fontWeight: 600}}>tok/s prompt</span>
        </div>
      </div>
      <div style={{height: font * 1.1, borderRadius: 99, background: 'rgba(255,255,255,0.07)', overflow: 'hidden', margin: `${font}px 0`}}>
        <div
          style={{
            width: `${(done / total) * 100}%`,
            height: '100%',
            borderRadius: 99,
            background: fast ? 'linear-gradient(90deg,#f97316,#facc15,#fff4b0)' : '#64748b',
          }}
        />
      </div>
      <div style={{display: 'flex', justifyContent: 'space-between', fontFamily: MONO, fontSize: font, color: '#94a3b8', fontVariantLigatures: 'none'}}>
        <span>
          {Math.round(done).toLocaleString('en-US')} / {total.toLocaleString('en-US')} tokens read
        </span>
        <span style={{color: finished ? (fast ? ACCENT : '#e2e8f0') : '#94a3b8'}}>
          {finished ? `✓ first token after ${fmtTime(full)}` : `${fmtTime(simT)} · ETA ${fmtTime(full)}`}
        </span>
      </div>
    </div>
  );
};

export const PrefillScene: React.FC<{
  wide: boolean;
  model: string;
  slowName: string;
  fastName: string;
  prefillTokens: number;
  slowPrefillRate: number;
  fastPrefillRate: number;
  footnote: string;
}> = (p) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  const font = p.wide ? 20 : 18;

  const fastFull = p.prefillTokens / p.fastPrefillRate;
  const speed = fastFull / (BARS_END - BARS_START); // time-lapse factor
  const simT = interpolate(t, [BARS_START, PREFILL_SCENE], [0, (PREFILL_SCENE - BARS_START) * speed], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const titleIn = spring({frame, fps, config: {damping: 200}, durationInFrames: 20});
  const lane1 = spring({frame: frame - 0.3 * fps, fps, config: {damping: 200}, durationInFrames: 20});
  const lane2 = spring({frame: frame - 0.45 * fps, fps, config: {damping: 200}, durationInFrames: 20});
  const verdict = interpolate(t, [BARS_END + 0.1, BARS_END + 0.6], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  const ratio = p.fastPrefillRate / p.slowPrefillRate;

  return (
    <AbsoluteFill style={{padding: p.wide ? 56 : 40, display: 'flex', flexDirection: 'column', gap: p.wide ? 34 : 28}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 20, opacity: titleIn, transform: `translateY(${(1 - titleIn) * 20}px)`}}>
        <Bolt size={p.wide ? 64 : 58} />
        <div>
          <div style={{fontSize: p.wide ? 40 : 34, fontWeight: 800, letterSpacing: -1}}>
            ① Reading a {Math.round(p.prefillTokens / 1000)}K-token prompt
          </div>
          <div style={{fontSize: p.wide ? 20 : 18, color: '#94a3b8', marginTop: 4}}>
            Same model · same prompt · same PC · {p.model}
          </div>
        </div>
        <div
          style={{
            marginLeft: 'auto',
            alignSelf: 'flex-start',
            fontFamily: MONO,
            fontSize: p.wide ? 18 : 15,
            color: '#94a3b8',
            border: '1px solid rgba(148,163,184,0.3)',
            borderRadius: 999,
            padding: '6px 14px',
            opacity: t > BARS_START ? 1 : 0,
            whiteSpace: 'nowrap',
          }}
        >
          ⏩ time-lapse {Math.round(speed)}×
        </div>
      </div>

      <div style={{flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: p.wide ? 30 : 26}}>
        <div style={{fontFamily: MONO, fontSize: font * 0.9, color: '#64748b', fontVariantLigatures: 'none'}}>
          ❯ [pasted repository: {p.prefillTokens.toLocaleString('en-US')} tokens of code] Find the bug in the interval merger and fix it.
        </div>
        <Lane name={p.slowName} fast={false} rate={p.slowPrefillRate} total={p.prefillTokens} simT={simT} font={font} appear={lane1} />
        <Lane name={p.fastName} fast rate={p.fastPrefillRate} total={p.prefillTokens} simT={simT} font={font} appear={lane2} />
        <div style={{display: 'flex', justifyContent: 'center', opacity: verdict, transform: `scale(${0.9 + verdict * 0.1})`}}>
          <div
            style={{
              fontSize: p.wide ? 44 : 36,
              fontWeight: 900,
              background: 'linear-gradient(180deg,#fff4b0,#facc15 50%,#f97316)',
              WebkitBackgroundClip: 'text',
              color: 'transparent',
            }}
          >
            {ratio.toFixed(0)}× faster prompt processing
          </div>
        </div>
      </div>
      <div style={{fontSize: p.wide ? 14 : 13, color: '#64748b', textAlign: 'center', lineHeight: 1.4}}>{p.footnote}</div>
    </AbsoluteFill>
  );
};
