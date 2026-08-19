/**
 * Text-to-speech for the sentence builder.
 *
 * v1 called `speechSynthesis.speak` directly with a hardcoded `en-US`. This wrapper
 * checks support first and cancels any in-flight utterance, so pressing Speak twice
 * does not queue two readings.
 */
export function isSpeechSupported(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

/**
 * Speak `text`, replacing anything currently being spoken.
 *
 * @returns `false` if speech synthesis is unavailable or the text was blank.
 */
export function speak(text: string, language = 'en-US'): boolean {
  const trimmed = text.trim();
  if (trimmed.length === 0 || !isSpeechSupported()) return false;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(trimmed);
  utterance.lang = language;
  window.speechSynthesis.speak(utterance);
  return true;
}
