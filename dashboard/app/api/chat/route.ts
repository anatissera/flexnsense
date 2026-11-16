import { SYSTEM_PROMPT, TOOLS } from "../../lib/chatContext";
import { clientKey, take, tooMany } from "../ratelimit";

/**
 * The assistant's text side, against whichever provider has a key here.
 *
 * Gemini publishes an OpenAI-compatible chat-completions endpoint, tool calls included, so the
 * only differences are the URL, the model and one OpenAI-only parameter. That makes the provider
 * a deployment decision rather than a code change, which matters because a public demonstration
 * runs on whoever's key the team has to spare.
 *
 * The voice side is NOT portable this way. `/api/realtime` mints an ephemeral secret and the
 * browser then speaks OpenAI's realtime event protocol directly; Google's Live API is a different
 * protocol, not a different base URL. With a Gemini key, text works and voice says why it cannot.
 */

/** Generous for a person asking questions, cheap for a loop. */
const LIMIT = 20;
const WINDOW_S = 60;

type Provider = { name: string; url: string; key: string; model: string; openAiOnly: boolean };

/** Gemini first when its key is present, so setting it is how you choose. */
function provider(): Provider | null {
  const gemini = process.env.GEMINI_API_KEY;
  if (gemini) {
    return {
      name: "Gemini",
      url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
      key: gemini,
      model: process.env.GEMINI_MODEL || "gemini-2.5-flash",
      openAiOnly: false,
    };
  }
  const openai = process.env.OPENAI_API_KEY;
  if (openai) {
    return {
      name: "OpenAI",
      url: "https://api.openai.com/v1/chat/completions",
      key: openai,
      model: process.env.OPENAI_MODEL || "gpt-6-luna",
      openAiOnly: true,
    };
  }
  return null;
}

export async function POST(req: Request) {
  const p = provider();
  if (!p) {
    return Response.json(
      { error: "No model key is set. Put GEMINI_API_KEY or OPENAI_API_KEY in the environment." },
      { status: 500 },
    );
  }

  const limit = take(`chat:${clientKey(req)}`, LIMIT, WINDOW_S);
  if (!limit.ok) return tooMany(limit.retryAfterS);

  let body: { messages?: unknown[]; context?: unknown };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  if (!Array.isArray(body.messages)) return Response.json({ error: "messages required" }, { status: 400 });

  const res = await fetch(p.url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${p.key}` },
    body: JSON.stringify({
      model: p.model,
      messages: [
        { role: "system", content: `${SYSTEM_PROMPT}\n\nCONTEXT:\n${JSON.stringify(body.context ?? {})}` },
        ...body.messages.slice(-40),
      ],
      tools: TOOLS,
      // gpt-6-luna rejects function tools on /v1/chat/completions unless reasoning is off. The
      // parameter is OpenAI's own, so it is not sent to a compatibility endpoint that never
      // promised to understand it.
      ...(p.openAiOnly ? { reasoning_effort: "none" } : {}),
    }),
  });
  const json = await res.json().catch(() => null);
  if (!res.ok || !json?.choices?.[0]?.message) {
    return Response.json(
      { error: json?.error?.message ?? `${p.name} error ${res.status} (model ${p.model})` },
      { status: 502 },
    );
  }
  return Response.json({ message: json.choices[0].message });
}
