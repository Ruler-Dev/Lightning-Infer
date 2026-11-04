export const PROMPT = 'Write a Python function that merges overlapping intervals, with a test.';

export const RESPONSE = `Sort by start, then sweep once and extend the last interval whenever the next one overlaps.

\`\`\`python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge overlapping [start, end] intervals in O(n log n)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def test_merge_intervals():
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]
\`\`\`

Sorting dominates, so the whole thing runs in O(n log n) time and O(n) extra space.`;

// Rough BPE-like split: words, runs of spaces/newlines, single punctuation.
export const TOKENS: string[] = RESPONSE.match(/\n|[ ]+|[A-Za-z_]+|\d+|[^\sA-Za-z_\d]/g) ?? [];

// Deterministic PRNG so every render is identical.
const mulberry32 = (a: number) => () => {
  a |= 0; a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** Emission time (seconds after first token) of each token at a mean rate with jitter. */
export const schedule = (rate: number, seed: number, jitter = 0.35): number[] => {
  const rnd = mulberry32(seed);
  const out: number[] = [];
  let t = 0;
  for (let i = 0; i < TOKENS.length; i++) {
    out.push(t);
    t += (1 / rate) * (1 + (rnd() * 2 - 1) * jitter);
  }
  // normalise so the mean rate is exact
  const scale = (TOKENS.length - 1) / rate / out[out.length - 1];
  return out.map((x) => x * scale);
};

const KW = new Set(['def', 'return', 'if', 'else', 'for', 'in', 'not', 'lambda', 'assert', 'list', 'int', 'max', 'sorted']);
export const colorFor = (tok: string, inCode: boolean, accent: string): string => {
  if (!inCode) return '#e5e7eb';
  if (KW.has(tok)) return accent;
  if (/^\d+$/.test(tok)) return '#93c5fd';
  if (/^[^\w\s]$/.test(tok)) return '#9ca3af';
  return '#e5e7eb';
};
