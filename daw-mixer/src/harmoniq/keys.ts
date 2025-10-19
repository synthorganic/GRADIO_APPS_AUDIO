export const HARMONIC_WHEEL_KEYS = [
  "C",
  "G",
  "D",
  "A",
  "E",
  "B",
  "F#",
  "Db",
  "Ab",
  "Eb",
  "Bb",
  "F"
];

const CAMEL0T_TO_WHEEL: Record<string, string> = {
  "1A": "Ab",
  "2A": "Eb",
  "3A": "Bb",
  "4A": "F",
  "5A": "C",
  "6A": "G",
  "7A": "D",
  "8A": "A",
  "9A": "E",
  "10A": "B",
  "11A": "F#",
  "12A": "Db",
  "1B": "B",
  "2B": "F#",
  "3B": "Db",
  "4B": "Ab",
  "5B": "Eb",
  "6B": "Bb",
  "7B": "F",
  "8B": "C",
  "9B": "G",
  "10B": "D",
  "11B": "A",
  "12B": "E"
};

const ENHARMONIC_EQUIVALENTS: Record<string, string> = {
  "G#": "Ab",
  "A#": "Bb",
  "C#": "Db",
  "D#": "Eb"
};

export function normalizeKeyForWheel(key?: string | null): string | null {
  if (!key) return null;
  const raw = key.trim();
  if (!raw) return null;

  const camelot = raw.toUpperCase();
  if (CAMEL0T_TO_WHEEL[camelot]) {
    return CAMEL0T_TO_WHEEL[camelot];
  }

  const normalized = raw
    .replace(/♯/g, "#")
    .replace(/♭/g, "b")
    .replace(/\s+/g, "")
    .toUpperCase();

  const noteMatch = normalized.match(/^([A-G])([#B]?)/);
  if (!noteMatch) {
    return null;
  }

  const [, base, accidental = ""] = noteMatch;
  if (accidental === "B") {
    const flatKey = `${base}b`;
    return (HARMONIC_WHEEL_KEYS.includes(flatKey) && flatKey) || null;
  }

  const note = `${base}${accidental}`;
  if (HARMONIC_WHEEL_KEYS.includes(note)) {
    return note;
  }

  return ENHARMONIC_EQUIVALENTS[note] ?? null;
}
