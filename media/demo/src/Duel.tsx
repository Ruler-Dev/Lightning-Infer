import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing} from 'remotion';
import {Bolt} from './Logo';
import {PROMPT, TOKENS, colorFor, schedule} from './tokens';
import {PrefillScene, PREFILL_SCENE} from './Prefill';

export type DuelProps = {
  layout: 'wide' | 'tall';
  model: string;
  slowName: string;
  slowRate: number;
  fastName: string;
  fastRate: number;
  prefillTokens: number;
  slowPrefillRate: number;
  fastPrefillRate: number;
  footnote: string;
  repo: string;
};

const BG = '#05070d';
const ACCENT = '#facc15';
const SLOW = '#94a3b8';
const MONO = '"JetBrains Mono", "Cascadia Code", Consolas, "SF Mono", monospace';
const SANS = 'Inter, "Segoe UI", system-ui, -apple-system, sans-serif';

// Timeline (seconds)
const PANELS_IN = 0.5;
const GEN_START = 1.4;
const OUTRO = 9.0;

type Engine = {
  name: string;
  rate: number;
  ttft: number;
  seed: number;
  color: string;
  fast: boolean;
};

const Panel: React.FC<{e: Engine; t: number; font: number; times: number[]}> = ({e, t, font, times}) => {
  const genT = t - GEN_START - e.ttft;
  let n = 0;
  if (genT >= 0) {
    while (n < times.length && times[n] <= genT) n++;
  }
  const done = n >= TOKENS.length;
  const thinking = t >= GEN_START && genT < 0;

  // live rate: tokens in the last second
  let rate = 0;
  if (genT > 0.15) {
    const w = Math.min(1, genT);
    const from = genT - w;
    const k = times.filter((x) => x > from && x <= genT).length;
    rate = done ? (TOKENS.length - 1) / times[times.length - 1] : k / w;
  }
  const elapsed = Math.max(0, t - GEN_START);
  const finishedAt = GEN_START + e.ttft + times[times.length - 1];
  const shownElapsed = done ? finishedAt - GEN_START : elapsed;

  // build coloured text
  let inCode = false;
  let fence = 0;
  const spans: React.ReactNode[] = [];
  for (let i = 0; i < n; i++) {
    const tok = TOKENS[i];
    if (tok === '`') {
      fence++;
      if (fence % 3 === 0) inCode = !inCode;
    }
    spans.push(
      <span key={i} style={{color: colorFor(tok, inCode, e.color)}}>
        {tok}
      </span>,
    );
  }
  const cursorOn = Math.floor(t * 2.5) % 2 === 0;

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 18,
        background: 'linear-gradient(180deg, #0d1220 0%, #080b14 100%)',
        border: `1.5px solid ${e.fast ? 'rgba(250,204,21,0.55)' : 'rgba(148,163,184,0.25)'}`,
        boxShadow: e.fast ? '0 0 60px rgba(250,204,21,0.12)' : 'none',
        overflow: 'hidden',
        minWidth: 0,
      }}
    >
      {/* title bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: `${font * 0.8}px ${font * 1.1}px`,
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          fontFamily: SANS,
        }}
      >
        <div style={{display: 'flex', gap: 7}}>
          {['#ef4444', '#f59e0b', '#22c55e'].map((c) => (
            <div key={c} style={{width: 11, height: 11, borderRadius: 6, background: c, opacity: 0.8}} />
          ))}
        </div>
        <div style={{fontSize: font * 1.25, fontWeight: 700, color: e.fast ? ACCENT : '#cbd5e1', marginLeft: 8}}>
          {e.fast ? '⚡ ' : ''}
          {e.name}
        </div>
        <div style={{marginLeft: 'auto', fontSize: font * 0.85, color: '#64748b', fontFamily: MONO}}>
          {thinking ? 'prefill…' : done ? 'done' : t >= GEN_START ? 'generating' : 'idle'}
        </div>
      </div>

      {/* stream */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column-reverse',
          overflow: 'hidden',
          padding: `${font * 0.9}px ${font * 1.1}px`,
        }}
      >
        <div
          style={{
            flexShrink: 0,
            minHeight: '100%',
            fontFamily: MONO,
            fontSize: font,
            lineHeight: 1.42,
            whiteSpace: 'pre-wrap',
            fontVariantLigatures: 'none',
            wordBreak: 'break-word',
          }}
        >
          <div style={{color: '#64748b', marginBottom: font * 0.8}}>
            <span style={{color: e.fast ? ACCENT : SLOW}}>❯ </span>
            {PROMPT}
          </div>
          {spans}
          {!done && t >= GEN_START && (
            <span
              style={{
                display: 'inline-block',
                width: font * 0.55,
                height: font * 1.1,
                verticalAlign: 'text-bottom',
                background: cursorOn ? (e.fast ? ACCENT : SLOW) : 'transparent',
              }}
            />
          )}
        </div>
      </div>

      {/* stats */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: font * 2,
          padding: `${font * 0.9}px ${font * 1.1}px ${font * 1.1}px`,
          borderTop: '1px solid rgba(255,255,255,0.07)',
          fontFamily: SANS,
        }}
      >
        <div>
          <div style={{fontSize: font * 0.75, letterSpacing: 2, color: '#64748b', fontWeight: 600}}>DECODE</div>
          <div style={{display: 'flex', alignItems: 'baseline', gap: 8}}>
            <span
              style={{
                fontSize: font * 3.4,
                fontWeight: 800,
                lineHeight: 1,
                color: e.fast ? ACCENT : '#e2e8f0',
                fontVariantNumeric: 'tabular-nums',
                textShadow: e.fast ? '0 0 30px rgba(250,204,21,0.45)' : 'none',
              }}
            >
              {rate > 0 ? rate.toFixed(0) : '—'}
            </span>
            <span style={{fontSize: font * 1.1, color: '#94a3b8', fontWeight: 600}}>tok/s</span>
          </div>
        </div>
        <div style={{flex: 1, paddingBottom: font * 0.5}}>
          <div style={{display: 'flex', justifyContent: 'space-between', fontSize: font * 0.85, color: '#94a3b8', fontFamily: MONO, marginBottom: 8}}>
            <span>
              {n}/{TOKENS.length} tokens
            </span>
            <span>{t >= GEN_START ? `${shownElapsed.toFixed(1)} s` : ''}</span>
          </div>
          <div style={{height: font * 0.55, borderRadius: 99, background: 'rgba(255,255,255,0.07)', overflow: 'hidden'}}>
            <div
              style={{
                width: `${(n / TOKENS.length) * 100}%`,
                height: '100%',
                borderRadius: 99,
                background: e.fast ? 'linear-gradient(90deg,#f97316,#facc15,#fff4b0)' : '#64748b',
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export const Duel: React.FC<DuelProps> = (p) => {
  const absFrame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const frame = absFrame - PREFILL_SCENE * fps;
  const t = frame / fps;
  const sceneMix = interpolate(t, [-0.4, 0], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const wide = p.layout === 'wide';
  const font = wide ? 15 : 14;
  const pad = wide ? 56 : 40;

  const slow: Engine = {name: p.slowName, rate: p.slowRate, ttft: 0.9, seed: 7, color: '#cbd5e1', fast: false};
  const fast: Engine = {name: p.fastName, rate: p.fastRate, ttft: 0.25, seed: 42, color: ACCENT, fast: true};
  const slowTimes = React.useMemo(() => schedule(slow.rate, slow.seed), [slow.rate]);
  const fastTimes = React.useMemo(() => schedule(fast.rate, fast.seed, 0.5), [fast.rate]);

  const titleIn = spring({frame, fps, config: {damping: 200}, durationInFrames: 20});
  const panelsIn = spring({frame: frame - PANELS_IN * fps, fps, config: {damping: 200}, durationInFrames: 25});
  const outro = interpolate(t, [OUTRO, OUTRO + 0.6], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const speedup = p.fastRate / p.slowRate;
  const counterPre = interpolate(t, [OUTRO + 0.3, OUTRO + 1.3], [1, p.fastPrefillRate / p.slowPrefillRate], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const counter = interpolate(t, [OUTRO + 0.3, OUTRO + 1.3], [1, speedup], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});

  return (
    <AbsoluteFill style={{background: BG, fontFamily: SANS, color: 'white'}}>
      {/* backdrop */}
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at 75% 0%, rgba(250,204,21,0.10), transparent 55%), radial-gradient(ellipse at 10% 100%, rgba(99,102,241,0.10), transparent 50%)',
        }}
      />
      <AbsoluteFill
        style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />

      {sceneMix < 1 && (
        <AbsoluteFill style={{opacity: 1 - sceneMix}}>
          <PrefillScene wide={wide} model={p.model} slowName={p.slowName} fastName={p.fastName} prefillTokens={p.prefillTokens} slowPrefillRate={p.slowPrefillRate} fastPrefillRate={p.fastPrefillRate} footnote={p.footnote} />
        </AbsoluteFill>
      )}
      {sceneMix > 0 && (
      <AbsoluteFill style={{padding: pad, display: 'flex', flexDirection: 'column', gap: wide ? 28 : 24, filter: `blur(${outro * 8}px)`, opacity: sceneMix * (1 - outro * 0.65)}}>
        {/* header */}
        <div style={{display: 'flex', alignItems: 'center', gap: 20, opacity: titleIn, transform: `translateY(${(1 - titleIn) * 20}px)`}}>
          <Bolt size={wide ? 64 : 58} />
          <div>
            <div style={{fontSize: wide ? 40 : 36, fontWeight: 800, letterSpacing: -1}}>
              ② Writing the answer
            </div>
            <div style={{fontSize: wide ? 20 : 19, color: '#94a3b8', marginTop: 4}}>
              {p.model} · single request · streaming decode · real time
            </div>
          </div>
        </div>

        {/* panels */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: wide ? 'row' : 'column',
            gap: wide ? 28 : 20,
            minHeight: 0,
            opacity: panelsIn,
            transform: `translateY(${(1 - panelsIn) * 30}px)`,
          }}
        >
          <Panel e={slow} t={t} font={font} times={slowTimes} />
          <Panel e={fast} t={t} font={font} times={fastTimes} />
        </div>

        <div style={{fontSize: wide ? 14 : 13, color: '#64748b', textAlign: 'center', lineHeight: 1.4}}>{p.footnote}</div>
      </AbsoluteFill>
      )}

      {/* outro */}
      {outro > 0 && (
        <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', opacity: outro}}>
          <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18, transform: `scale(${0.92 + outro * 0.08})`}}>
            <Bolt size={wide ? 150 : 140} glow={1.6} />
            <div style={{fontSize: wide ? 34 : 30, fontWeight: 700}}>Lightning-Infer vs {p.slowName}</div>
            <div style={{display: 'flex', gap: wide ? 90 : 50, marginTop: 6}}>
              {[
                {v: counterPre, l: 'faster prompt processing', d: `${p.fastPrefillRate.toLocaleString('en-US')} vs ${p.slowPrefillRate.toLocaleString('en-US')} tok/s`},
                {v: counter, l: 'faster decode', d: `${p.fastRate} vs ${p.slowRate} tok/s`},
              ].map((x) => (
                <div key={x.l} style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                  <div
                    style={{
                      fontSize: wide ? 130 : 104,
                      fontWeight: 900,
                      letterSpacing: -4,
                      lineHeight: 1,
                      background: 'linear-gradient(180deg,#fff4b0,#facc15 50%,#f97316)',
                      WebkitBackgroundClip: 'text',
                      color: 'transparent',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {x.v.toFixed(1)}×
                  </div>
                  <div style={{fontSize: wide ? 30 : 26, fontWeight: 700, marginTop: 8}}>{x.l}</div>
                  <div style={{fontSize: wide ? 22 : 19, color: '#94a3b8', marginTop: 4}}>{x.d}</div>
                </div>
              ))}
            </div>
            <div style={{fontSize: wide ? 20 : 18, color: '#64748b'}}>{p.model}</div>
            <div
              style={{
                marginTop: 14,
                padding: '12px 26px',
                borderRadius: 999,
                border: '1.5px solid rgba(250,204,21,0.5)',
                fontFamily: MONO,
                fontSize: wide ? 22 : 20,
                fontVariantLigatures: 'none',
                color: ACCENT,
              }}
            >
              {p.repo}
            </div>
          </div>
          <div style={{position: 'absolute', bottom: pad * 0.6, fontSize: 14, color: '#64748b', textAlign: 'center', width: '85%'}}>{p.footnote}</div>
        </AbsoluteFill>
      )}
      <div style={{position: 'absolute', width, height, pointerEvents: 'none'}} />
    </AbsoluteFill>
  );
};
