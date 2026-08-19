import { describe, expect, it, vi } from 'vitest';
import { SentenceBuffer } from './sentenceBuffer.js';

describe('SentenceBuffer', () => {
  it('starts empty', () => {
    const buffer = new SentenceBuffer();
    expect(buffer.text).toBe('');
    expect(buffer.isEmpty).toBe(true);
  });

  it('appends letters in order', () => {
    const buffer = new SentenceBuffer();
    buffer.append('H');
    buffer.append('I');
    expect(buffer.text).toBe('HI');
    expect(buffer.isEmpty).toBe(false);
  });

  it('appends spaces', () => {
    const buffer = new SentenceBuffer();
    buffer.append('A');
    buffer.appendSpace();
    buffer.append('B');
    expect(buffer.text).toBe('A B');
  });

  it('removes the last character on backspace', () => {
    const buffer = new SentenceBuffer();
    buffer.append('ABC');
    buffer.backspace();
    expect(buffer.text).toBe('AB');
  });

  it('ignores backspace when already empty', () => {
    const buffer = new SentenceBuffer();
    expect(() => buffer.backspace()).not.toThrow();
    expect(buffer.text).toBe('');
  });

  it('clears everything', () => {
    const buffer = new SentenceBuffer();
    buffer.append('HELLO');
    buffer.clear();
    expect(buffer.text).toBe('');
  });

  it('ignores an empty append', () => {
    const buffer = new SentenceBuffer();
    const listener = vi.fn();
    buffer.subscribe(listener);
    buffer.append('');
    expect(listener).not.toHaveBeenCalled();
  });

  it('notifies subscribers on every mutation', () => {
    const buffer = new SentenceBuffer();
    const listener = vi.fn();
    buffer.subscribe(listener);

    buffer.append('A');
    buffer.appendSpace();
    buffer.backspace();
    buffer.clear();

    expect(listener.mock.calls.map((call) => call[0] as string)).toEqual(['A', 'A ', 'A', '']);
  });

  it('does not notify when a no-op leaves the text unchanged', () => {
    const buffer = new SentenceBuffer();
    const listener = vi.fn();
    buffer.subscribe(listener);
    buffer.backspace();
    buffer.clear();
    expect(listener).not.toHaveBeenCalled();
  });

  it('stops notifying after unsubscribe', () => {
    const buffer = new SentenceBuffer();
    const listener = vi.fn();
    const unsubscribe = buffer.subscribe(listener);
    buffer.append('A');
    unsubscribe();
    buffer.append('B');
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
