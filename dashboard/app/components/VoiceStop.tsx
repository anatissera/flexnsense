"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { fireEmergencyStop } from "../lib/estop";

/**
 * "Stop" out loud, in the language the person actually speaks.
 *
 * A patient with a finger tied to a robot arm has one free hand and may not have it near a
 * mouse. A clinician standing behind them has the same problem. So the stop is also a word, and
 * it has to be a word they would actually shout: this is a Swiss hackathon in a canton where the
 * instruction might come in German, French or English, and the team speaks Spanish.
 *
 * Browser speech recognition takes one language at a time, so the picker sets the recogniser. The
 * matcher is deliberately not restricted to it: a recogniser running in German will still
 * transcribe an English "stop" close enough, and accepting every language's stop words costs
 * nothing. A false positive stops a set that can be restarted. A missed stop is the failure that
 * matters, which is the same reason the force gate releases on a reading it cannot trust.
 *
 * What this is not: a safety system. It runs in a browser tab, needs a network for recognition in
 * most browsers, and is not what keeps a real finger safe. On the rig that is the force gate, the
 * interlock and Station's own current limits, none of which need this page to exist.
 */

/** The minimum of the Web Speech API this uses. It is not in the standard DOM typings. */
type SpeechResult = { 0: { transcript: string }; isFinal: boolean; length: number };
type SpeechEvent = { resultIndex: number; results: { length: number } & Record<number, SpeechResult> };
type Recogniser = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((e: SpeechEvent) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
};
type RecogniserCtor = new () => Recogniser;

function recogniserCtor(): RecogniserCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { SpeechRecognition?: RecogniserCtor; webkitSpeechRecognition?: RecogniserCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export const LANGUAGES = [
  { code: "en-US", label: "English" },
  { code: "es-ES", label: "Español" },
  { code: "de-DE", label: "Deutsch" },
  { code: "fr-FR", label: "Français" },
] as const;

/**
 * Every language's stop words, matched whatever the recogniser is set to.
 *
 * "para" is in the Spanish list knowing it is also an ordinary preposition. Shouted at a machine
 * holding somebody's finger it means stop, and a spurious stop costs one restarted set.
 */
const STOP_WORDS = [
  "stop", "stop it", "halt",
  "para", "pare", "parar", "alto", "basta", "detener",
  "stopp", "anhalten", "aufhoren", "aufhalten",
  "arrete", "arreter", "halte",
];

/** Lowercase, without accents, punctuation reduced to spaces: "¡Pará!" and "arrête" both land. */
export function normalise(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** The stop word heard in `text`, or null. Whole words only, so "stopper" does not fire. */
export function heardStop(text: string): string | null {
  const words = normalise(text).split(" ");
  for (const phrase of STOP_WORDS) {
    const parts = phrase.split(" ");
    for (let i = 0; i + parts.length <= words.length; i++) {
      if (parts.every((p, k) => words[i + k] === p)) return phrase;
    }
  }
  return null;
}

/** The picker's default: the browser's own language when it is one we listen in. */
function defaultLang(): string {
  if (typeof navigator === "undefined") return "en-US";
  const prefix = navigator.language?.slice(0, 2).toLowerCase();
  return LANGUAGES.find((l) => l.code.slice(0, 2) === prefix)?.code ?? "en-US";
}

export default function VoiceStop() {
  const [on, setOn] = useState(false);
  const [lang, setLang] = useState(defaultLang);
  const [heard, setHeard] = useState("");
  const [fired, setFired] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Whether this browser has speech recognition at all. Read through useSyncExternalStore because
  // the answer does not exist on the server: the server snapshot assumes yes, the browser corrects
  // it, and that disagreement is the hook's whole purpose rather than a hydration bug.
  const supported = useSyncExternalStore(() => () => {}, () => recogniserCtor() !== null, () => true);
  const recRef = useRef<Recogniser | null>(null);
  const firedTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(firedTimer.current), []);

  useEffect(() => {
    if (!on) return;
    // Unreachable while the button is rendered, which only happens when there is a constructor.
    const Ctor = recogniserCtor();
    if (!Ctor) return;
    let stopped = false;
    const rec = new Ctor();
    recRef.current = rec;
    rec.lang = lang;
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onresult = (e) => {
      let text = "";
      for (let i = e.resultIndex; i < e.results.length; i++) text += ` ${e.results[i][0].transcript}`;
      const trimmed = text.trim();
      if (trimmed) setHeard(trimmed.slice(-60));
      // Interim results are matched too: waiting for the recogniser to decide a phrase is final
      // adds most of a second to a stop, and the word is unambiguous enough to act on early.
      const word = heardStop(trimmed);
      if (word) {
        fireEmergencyStop(`heard "${word}"`);
        setFired(word);
        clearTimeout(firedTimer.current);
        firedTimer.current = setTimeout(() => setFired(null), 8000);
      }
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        setError("The browser would not give this page the microphone.");
        setOn(false);
      } else if (e.error !== "no-speech" && e.error !== "aborted") {
        setError(`Speech recognition error: ${e.error}`);
      }
    };
    // Chrome ends the session on its own after a silence. While the control is on, start again.
    rec.onend = () => {
      if (stopped) return;
      try {
        rec.start();
      } catch {
        /* already starting; the next onend will try again */
      }
    };

    try {
      rec.start();
    } catch {
      // start() throws synchronously only when this recogniser is already running, which the
      // restart in onend can cause and which is harmless. Everything that actually goes wrong,
      // a refused microphone or no network for the recognition service, arrives at onerror.
    }

    return () => {
      stopped = true;
      rec.onend = null;
      rec.onresult = null;
      rec.onerror = null;
      try {
        rec.abort();
      } catch {
        /* nothing left to abort */
      }
      recRef.current = null;
      setHeard("");
    };
  }, [on, lang]);

  if (!supported) {
    return (
      <span className="t-small faint" title="Chrome, Edge and Safari have it; Firefox does not.">
        Voice stop needs a browser with speech recognition
      </span>
    );
  }

  return (
    <span className="flex items-center gap-2">
      {fired && (
        <span className="t-small font-semibold" style={{ color: "var(--danger)" }}>
          Heard &ldquo;{fired}&rdquo;, stopped
        </span>
      )}
      {!fired && on && heard && <span className="t-small faint max-w-[160px] truncate">{heard}</span>}
      {error && <span className="t-small" style={{ color: "var(--danger)" }}>{error}</span>}
      <select
        className="btn btn-sm"
        value={lang}
        onChange={(e) => setLang(e.target.value)}
        title="Which language the recogniser listens in. Stop words from all four are accepted whichever is chosen."
      >
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code} className="text-black">{l.label}</option>
        ))}
      </select>
      <button
        className={on ? "btn btn-sm btn-primary" : "btn btn-sm"}
        aria-pressed={on}
        onClick={() => { setError(null); setOn(!on); }}
        title={
          on
            ? "Listening for stop, para, stopp, arrête and the rest. Switch off to release the microphone."
            : "Listen for the word stop in English, Spanish, German or French, and abort the set when it is heard."
        }
      >
        {on ? "Listening for “stop”" : "Voice stop"}
      </button>
    </span>
  );
}
