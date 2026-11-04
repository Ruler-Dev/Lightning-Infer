import React from 'react';

export const Bolt: React.FC<{size: number; glow?: number}> = ({size, glow = 1}) => (
  <svg viewBox="0 0 512 512" width={size} height={size}>
    <defs>
      <linearGradient id="lbg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#151a2e" />
        <stop offset="1" stopColor="#05070d" />
      </linearGradient>
      <linearGradient id="lbolt" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor="#fff4b0" />
        <stop offset="0.45" stopColor="#facc15" />
        <stop offset="1" stopColor="#f97316" />
      </linearGradient>
      <radialGradient id="lglow" cx="0.5" cy="0.5" r="0.5">
        <stop offset="0" stopColor="#facc15" stopOpacity={0.45 * glow} />
        <stop offset="1" stopColor="#facc15" stopOpacity="0" />
      </radialGradient>
    </defs>
    <rect x="16" y="16" width="480" height="480" rx="112" fill="url(#lbg)" />
    <rect x="16" y="16" width="480" height="480" rx="112" fill="none" stroke="#facc15" strokeOpacity="0.35" strokeWidth="4" />
    <g stroke="#facc15" strokeOpacity="0.28" strokeWidth="10" strokeLinecap="round">
      <path d="M176 64v40M256 64v40M336 64v40M176 408v40M256 408v40M336 408v40" />
      <path d="M64 176h40M64 256h40M64 336h40M408 176h40M408 256h40M408 336h40" />
    </g>
    <rect x="120" y="120" width="272" height="272" rx="48" fill="none" stroke="#facc15" strokeOpacity="0.22" strokeWidth="6" />
    <circle cx="256" cy="256" r="170" fill="url(#lglow)" />
    <path d="M286 84 L150 284 H238 L214 428 L362 220 H270 L306 84 Z" fill="url(#lbolt)" stroke="#fff7cc" strokeWidth="6" strokeLinejoin="round" />
  </svg>
);
