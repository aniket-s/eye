/**
 * Dictionary vocabulary, carried over from v1 unchanged.
 *
 * ⚠️ None of the referenced images exist. Every entry falls through to the 🤟
 * placeholder — the Dictionary is currently 100% non-functional (docs/AUDIT.md, A1).
 * Phase 2 replaces these with CC0 photographs from the Roboflow ASL Letters set for
 * A–Z, and CC BY 4.0 clips from PopSign for words.
 */

export interface SignCategory {
  readonly id: string;
  readonly label: string;
  readonly signs: readonly string[];
}

export const CATEGORIES: readonly SignCategory[] = [
  {
    id: 'alphabets',
    label: '🔤 Alphabets',
    signs: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
  },
  {
    id: 'numbers',
    label: '🔢 Numbers',
    signs: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
  },
  {
    id: 'greetings',
    label: '👋 Greetings',
    signs: [
      'hello',
      'goodbye',
      'thank you',
      'please',
      'sorry',
      'yes',
      'no',
      'welcome',
      'good morning',
      'good night',
    ],
  },
  {
    id: 'family',
    label: '👨‍👩‍👧 Family',
    signs: [
      'mom',
      'dad',
      'sister',
      'brother',
      'aunt',
      'uncle',
      'son',
      'daughter',
      'grandma',
      'grandpa',
      'nephew',
      'niece',
      'cousin',
    ],
  },
  {
    id: 'directions',
    label: '🧭 Directions',
    signs: ['up', 'down', 'left', 'right', 'near', 'far', 'stop'],
  },
  {
    id: 'animals',
    label: '🐾 Animals',
    signs: ['dog', 'cat', 'fish', 'rabbit', 'bird', 'horse', 'lion'],
  },
  {
    id: 'sports',
    label: '⚽ Sports',
    signs: ['tennis', 'football', 'basketball', 'swim', 'run'],
  },
  {
    id: 'locations',
    label: '📍 Locations',
    signs: ['home', 'school', 'hospital', 'park', 'market', 'neighbourhood'],
  },
  {
    id: 'colors',
    label: '🎨 Colors',
    signs: ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink'],
  },
  {
    id: 'feelings',
    label: '😊 Feelings',
    signs: ['happy', 'sad', 'angry', 'scared', 'tired', 'sick', 'love', 'pain'],
  },
  {
    id: 'food',
    label: '🍎 Food & Drink',
    signs: ['water', 'eat', 'drink', 'milk', 'bread', 'apple', 'rice'],
  },
];

/** Image formats tried in order before falling back to the placeholder. */
export const IMAGE_FORMATS = ['jpg', 'png', 'webp'] as const;

/** Path to a sign's illustration, for a given format. */
export function signImagePath(categoryId: string, sign: string, format: string): string {
  return `${categoryId}/${sign.toLowerCase().replace(/ /g, '_')}.${format}`;
}
