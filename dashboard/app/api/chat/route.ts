import { SYSTEM_PROMPT, TOOLS } from "../../lib/chatContext";
import { clientKey, take, tooMany } from "../ratelimit";

/** Generous for a person asking questions, cheap for a loop. */
const LIMIT = 20;
const WINDOW_S = 60;

export async function POST(req: Request) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return Response.json({ error: "OPENAI_API_KEY is not set in .env" }, { status: 500 });

  const limit = take(`chat:${clientKey(req)}`, LIMIT, WINDOW_S);
  if (!limit.ok) return tooMany(limit.retryAfterS);

  let body: { messages?: unknown[]; context?: unknown };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  if (!Array.isArray(body.messages)) return Response.json({ error: "messages required" }, { status: 400 });

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || "gpt-6-luna",
      messages: [
        { role: "system", content: `${SYSTEM_PROMPT}\n\nCONTEXT:\n${JSON.stringify(body.context ?? {})}` },
        ...body.messages.slice(-40),
      ],
      tools: TOOLS,
      // gpt-6-luna rejects function tools on /v1/chat/completions unless reasoning is off.
      reasoning_effort: "none",
    }),
  });
  const json = await res.json().catch(() => null);
  if (!res.ok || !json?.choices?.[0]?.message) {
    return Response.json({ error: json?.error?.message ?? `OpenAI error ${res.status}` }, { status: 502 });
  }
  return Response.json({ message: json.choices[0].message });
}
