/**
 * Build the word list the app ships for word correction.
 *
 * Two sources, because neither alone is enough:
 *
 * - **`an-array-of-english-words`** (MIT, derived from SCOWL) decides what *is* a word.
 *   275,000 of them, which is far too many to offer as suggestions: `aalii` and `aasvogel`
 *   would sit in the candidate list competing with `also` and `ask`.
 * - **`wordfreq`** (Python, Apache-2.0) decides which of them anyone actually uses.
 *
 * The output is one word per line in descending frequency order, so a consumer gets a
 * usable prior for free — the line number *is* the rank, with no numbers to store or
 * parse. `docs/DEPLOY.md` counts the bytes; this is around 200 KB, ~70 KB over the wire.
 *
 * Run with `npm run build:lexicon`. The result is committed, so neither dependency is
 * needed to build or run the app.
 */

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = join(ROOT, 'packages/web/public/lexicon/en.txt');

/**
 * How many words to keep.
 *
 * Coverage of running English text climbs steeply to about 20,000 words and then flattens;
 * past that, each addition is likelier to be a false friend for a misspelling than a word
 * the user wanted. 30,000 keeps proper nouns and inflected forms without dragging in the
 * long tail of the dictionary.
 */
const KEEP = 30_000;

/** Zipf frequency below which a word is too rare to be a helpful suggestion. */
const MIN_ZIPF = 2.0;

const MIN_LENGTH = 2;
const MAX_LENGTH = 20;

function loadDictionary() {
  const path = join(ROOT, 'node_modules/an-array-of-english-words/index.json');
  let raw;
  try {
    raw = readFileSync(path, 'utf8');
  } catch {
    throw new Error(
      'an-array-of-english-words is not installed. Run `npm install` first — it is a ' +
        'devDependency, needed only to rebuild the lexicon.',
    );
  }
  return JSON.parse(raw);
}

/**
 * Score every word by frequency, in one Python call.
 *
 * Per-word subprocesses would take hours. This hands the whole list over on stdin and
 * reads the scores back.
 */
function frequencies(words) {
  const script = `
import sys
from wordfreq import zipf_frequency
out = []
for line in sys.stdin.read().split("\\n"):
    if line:
        out.append(f"{zipf_frequency(line, 'en'):.3f}")
sys.stdout.write("\\n".join(out))
`;
  const result = execFileSync('python3', ['-c', script], {
    input: words.join('\n'),
    maxBuffer: 256 * 1024 * 1024,
    encoding: 'utf8',
  });
  return result.split('\n').map(Number);
}

const dictionary = loadDictionary();

// Fingerspelling produces letters and nothing else — no apostrophes, no hyphens, no
// accents. A candidate the user cannot physically spell is noise in the ranking.
const spellable = dictionary.filter(
  (word) =>
    word.length >= MIN_LENGTH && word.length <= MAX_LENGTH && /^[a-z]+$/.test(word.toLowerCase()),
);

console.log(
  `${dictionary.length.toLocaleString()} words in, ${spellable.length.toLocaleString()} spellable`,
);

const scores = frequencies(spellable);
const ranked = spellable
  .map((word, index) => ({ word: word.toLowerCase(), zipf: scores[index] ?? 0 }))
  .filter((entry) => entry.zipf >= MIN_ZIPF)
  .sort((a, b) => b.zipf - a.zipf);

// The source list has both `Word` and `word` in places; lower-casing merges them.
const seen = new Set();
const kept = [];
for (const entry of ranked) {
  if (seen.has(entry.word)) continue;
  seen.add(entry.word);
  kept.push(entry.word);
  if (kept.length >= KEEP) break;
}

mkdirSync(dirname(OUTPUT), { recursive: true });
writeFileSync(OUTPUT, `${kept.join('\n')}\n`, 'utf8');

const bytes = Buffer.byteLength(kept.join('\n'));
console.log(
  `Wrote ${kept.length.toLocaleString()} words to packages/web/public/lexicon/en.txt ` +
    `(${(bytes / 1024).toFixed(0)} KB)`,
);
console.log(`  most common: ${kept.slice(0, 8).join(', ')}`);
console.log(`  least common: ${kept.slice(-8).join(', ')}`);
