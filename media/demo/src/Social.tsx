import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Bolt} from './Logo';

const SANS = 'Inter, "Segoe UI", system-ui, sans-serif';
const MONO = '"JetBrains Mono", "Cascadia Code", Consolas, monospace';

export const LogoStill: React.FC = () => (
  <AbsoluteFill style={{background: 'transparent', alignItems: 'center', justifyContent: 'center'}}>
    <Bolt size={512} />
  </AbsoluteFill>
);

const Stat: React.FC<{v: string; l: string}> = ({v, l}) => (
  <div>
    <div style={{fontSize: 46, fontWeight: 800, color: '#facc15', lineHeight: 1}}>{v}</div>
    <div style={{fontSize: 18, color: '#94a3b8', marginTop: 6}}>{l}</div>
  </div>
);

export const SocialCard: React.FC = () => (
  <AbsoluteFill style={{background: '#05070d', fontFamily: SANS, color: 'white'}}>
    <AbsoluteFill
      style={{
        background:
          'radial-gradient(ellipse at 85% 10%, rgba(250,204,21,0.16), transparent 55%), radial-gradient(ellipse at 0% 100%, rgba(99,102,241,0.14), transparent 55%)',
      }}
    />
    <AbsoluteFill
      style={{
        backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
      }}
    />
    <AbsoluteFill style={{padding: 72, display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 56}}>
      <Bolt size={300} glow={1.4} />
      <div style={{display: 'flex', flexDirection: 'column', gap: 18}}>
        <div style={{fontSize: 76, fontWeight: 900, letterSpacing: -2.5, lineHeight: 1}}>Lightning-Infer</div>
        <div style={{fontSize: 28, color: '#cbd5e1', lineHeight: 1.3, maxWidth: 760}}>
          A 70&nbsp;GB MoE model, split across two different consumer GPUs, running like a small one.
        </div>
        <div style={{display: 'flex', gap: 48, marginTop: 16}}>
          <Stat v="145" l="tok/s peak decode" />
          <Stat v="80–90" l="tok/s sustained" />
          <Stat v="512K" l="context" />
        </div>
        <div style={{fontFamily: MONO, fontSize: 20, color: '#64748b', marginTop: 10}}>C++ · CUDA · Qwen3.8-Flash-Next NVFP4 · Windows</div>
      </div>
    </AbsoluteFill>
  </AbsoluteFill>
);
