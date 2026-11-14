import { LANGUAGE_RULE, REALTIME_TOOLS } from "../../lib/chatContext";

// Mints a short-lived client secret so the browser can open a WebRTC session directly with
// OpenAI Realtime without ever seeing OPENAI_API_KEY.
export async function POST() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return Response.json({ error: "OPENAI_API_KEY is not set in .env" }, { status: 500 });

  const instructions = `You are a voice assistant embedded in a flexor-tendon recovery simulator (3D MyoHand + curl stress-test analysis).
Keep spoken answers short (1-3 sentences). Call get_scene_context before answering questions about results or numbers, and answer only from it. All data is simulated; no clinical advice.
When the user asks to see, find, or learn about a muscle/tendon or its properties, call show_muscle_info (properties) or highlight_muscles (just show), show_dashboard_section to scroll to and highlight the dashboard section for a metric, clear_highlights to reset, then briefly explain.
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
