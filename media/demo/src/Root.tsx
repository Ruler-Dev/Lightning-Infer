import React from 'react';
import {Composition, Still} from 'remotion';
import {Duel, DuelProps} from './Duel';
import {LogoStill, SocialCard} from './Social';

const FPS = 30;
const SECONDS = 18;

// Simulated playback: edit these numbers to match your own measurements.
const base: Omit<DuelProps, 'layout'> = {
  model: 'Qwen3.8-Flash-Next (NVFP4)',
  slowName: 'llama.cpp',
  slowRate: 10,
  fastName: 'Lightning-Infer',
  fastRate: 145,
  prefillTokens: 50000,
  slowPrefillRate: 300,
  fastPrefillRate: 2400,
  footnote:
    'Simulated playback. Measured on the same PC (RTX PRO 4000 Blackwell + RTX 5080): llama.cpp 10 tok/s decode; Lightning-Infer 145 tok/s peak decode (80–90 sustained) and a 50K-token prompt read in ~21 s. llama.cpp prompt-processing rate is an estimate.',
  repo: 'github.com/Ruler-Dev/Lightning-Infer',
};

export const Root: React.FC = () => (
  <>
    <Composition id="DuelWide" component={Duel} durationInFrames={FPS * SECONDS} fps={FPS} width={1920} height={1080} defaultProps={{...base, layout: 'wide' as const}} />
    <Composition id="DuelTall" component={Duel} durationInFrames={FPS * SECONDS} fps={FPS} width={1080} height={1350} defaultProps={{...base, layout: 'tall' as const}} />
    <Still id="Logo" component={LogoStill} width={512} height={512} />
    <Still id="Social" component={SocialCard} width={1280} height={640} />
  </>
);
