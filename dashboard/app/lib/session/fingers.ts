/**
 * Finger names, and the one place the two vocabularies are reconciled.
 *
 * Mirror of `session/fingers.py`. The tracker speaks "pinky", the clinical side (curlsim's
 * `Finger`, a prescription) speaks "little". The session layer canonicalises on "little".
 */
export const FINGERS = ["thumb", "index", "middle", "ring", "little"] as const;
export type SessionFinger = (typeof FINGERS)[number];

const ALIASES: Record<string, SessionFinger> = {
  pinky: "little",
  fifth: "little",
  "little finger": "little",
};

/** The canonical name for `name`, accepting either vocabulary. Throws on anything else. */
export function normaliseFinger(name: string): SessionFinger {
  const key = name.trim().toLowerCase();
  if ((FINGERS as readonly string[]).includes(key)) return key as SessionFinger;
  if (key in ALIASES) return ALIASES[key];
  throw new Error(
    `unknown finger ${JSON.stringify(name)}; known names are ${JSON.stringify(FINGERS)} ` +
      `(also accepted: ${JSON.stringify(Object.keys(ALIASES).sort())})`,
  );
}
