import { LANGUAGE_RULE, REALTIME_TOOLS } from "../../lib/chatContext";
import { clientKey, take, tooMany } from "../ratelimit";

/**
 * Tighter than the text route, because each secret buys a voice session that bills by the minute
 * and runs outside our reach once handed over. A person starts a voice session a few times in a
 * visit; nobody needs six a minute.
 */
const LIMIT = 6;
const WINDOW_S = 60;

// Mints a short-lived client secret so the browser can open a WebRTC session directly with
// OpenAI Realtime without ever seeing OPENAI_API_KEY.
export async function POST(req: Request) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    // The text route runs on a Gemini key too; this one cannot. The browser speaks OpenAI's
    // realtime event protocol directly, and Google's Live API is a different protocol rather
    // than a different base URL, so it would be a new client and not a new key.
    return Response.json(
      {
        error: process.env.GEMINI_API_KEY
          ? "Voice needs an OpenAI key. This deployment has a Gemini key, which the text assistant uses; the voice session speaks OpenAI's realtime protocol and has no Gemini equivalent here."
          : "OPENAI_API_KEY is not set in the environment.",
      },
      { status: 500 },
    );
  }

  const limit = take(`realtime:${clientKey(req)}`, LIMIT, WINDOW_S);
  if (!limit.ok) return tooMany(limit.retryAfterS);

  const instructions = `You are a voice assistant embedded in a flexor-tendon recovery simulator (3D MyoHand + curl stress-test analysis).
Keep spoken answers short (1-3 sentences). Call get_scene_context before answering questions about results or numbers, and answer only from it. All data is simulated; no clinical advice.
When the user asks to see, find, or learn about a muscle/tendon or its properties, call show_muscle_info (properties) or highlight_muscles (just show), show_dashboard_section to scroll to and highlight the dashboard section for a metric, clear_highlights to reset, tour_muscles to automatically step through every muscle popup, then briefly explain. Whenever your answer is about, or names, a specific muscle or tendon (even if the user did not ask to see it), also call highlight_muscles or show_muscle_info for the relevant muscle(s) so it lights up in the 3D view; skip it only when the answer isn't about a particular muscle.
FDP = deep flexor, FDS = superficial flexor, digits 2-5 = index, middle, ring, little. ${LANGUAGE_RULE}`;

  const res = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      session: {
        type: "realtime",
        model: process.env.OPENAI_REALTIME_MODEL || "gpt-realtime",
        instructions,
        tools: REALTIME_TOOLS,
        tool_choice: "auto",
        audio: {
          input: { transcription: { model: "gpt-4o-transcribe" }, turn_detection: { type: "semantic_vad" } },
          output: { voice: "marin" },
        },
      },
    }),
  });
  const json = await res.json().catch(() => null);
  const secret = json?.value ?? json?.client_secret?.value;
  if (!res.ok || !secret) {
    return Response.json({ error: json?.error?.message ?? `OpenAI error ${res.status}` }, { status: 502 });
  }
  return Response.json({ secret, model: process.env.OPENAI_REALTIME_MODEL || "gpt-realtime" });
}
