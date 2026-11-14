"use client";

import { useEffect, useRef, useState } from "react";
import Markdown from "./Markdown";

type ToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };
type Msg =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };

const SUGGESTIONS = [
  "Which tendon is recovering slowest?",
  "Highlight the FDP of the index finger",
  "Why is the ring finger's index low?",
];

export default function ChatPanel({
  context, onTool,
}: { context: unknown; onTool: (name: string, args: Record<string, unknown>) => string }) {
  const [open, setOpen] = useState(true);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef(context);
  useEffect(() => { ctxRef.current = context; }, [context]);
  const onToolRef = useRef(onTool);
  useEffect(() => { onToolRef.current = onTool; }, [onTool]);
  const [live, setLive] = useState({ user: "", assistant: "" });
  const [dbg, setDbg] = useState({ events: 0, last: "", level: 0 });
  const [voice, setVoice] = useState<"off" | "connecting" | "live">("off");
  const rtc = useRef<{ pc: null; stream: MediaStream; audio: null; stop: () => void } | null>(null);
  useEffect(() => () => stopVoice(), []);

  function stopVoice() {
    const r = rtc.current;
    if (!r) return;
    rtc.current = null;
    r.stop();
    setVoice("off");
  }

  // Voice over a WebSocket (24 kHz PCM16 both ways). Works the same in every browser, with no WebRTC/ICE.
  async function startVoice() {
    setError("");
    setVoice("connecting");
    setDbg({ events: 0, last: "starting", level: 0 });
    const ac = new AudioContext(); // created inside the click gesture
    void ac.resume();
    let stream: MediaStream | null = null;
    let ws: WebSocket | null = null;
    let node: AudioWorkletNode | null = null;
    const playing = new Set<AudioBufferSourceNode>();
    let playAt = 0;
    const cleanup = () => { stream?.getTracks().forEach((t) => t.stop()); node?.disconnect(); playing.forEach((n) => { try { n.stop(); } catch {} }); try { ws?.close(); } catch {} void ac.close().catch(() => {}); };
    try {
      const sess = await fetch("/api/realtime", { method: "POST" });
      const sj = await sess.json().catch(() => ({}));
      if (!sess.ok) throw new Error(sj.error ?? `Voice session failed (${sess.status})`);
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });

      const url = URL.createObjectURL(new Blob([`registerProcessor("cap", class extends AudioWorkletProcessor { process(i){ const c=i[0][0]; if(c) this.port.postMessage(c.slice(0)); return true; } });`], { type: "application/javascript" }));
      await ac.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      ws = new WebSocket(`wss://api.openai.com/v1/realtime?model=${encodeURIComponent(sj.model)}`, ["realtime", `openai-insecure-api-key.${sj.secret}`]);
      const sock = ws;
      const send = (o: unknown) => { if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify(o)); };
      rtc.current = { pc: null, stream, audio: null, stop: cleanup };

      // Capture: resample the mic from the context rate to 24 kHz, PCM16, base64.
      const ratio = ac.sampleRate / 24000;
      let carry = 0;
      let peak = 0;
      let lastUi = 0;
      node = new AudioWorkletNode(ac, "cap");
      node.port.onmessage = (m) => {
        const x = m.data as Float32Array;
        const out: number[] = [];
        for (; carry < x.length; carry += ratio) out.push(x[Math.floor(carry)]);
        carry -= x.length;
        const pcm = new Int16Array(out.length);
        for (let i = 0; i < out.length; i++) {
          const v = Math.max(-1, Math.min(1, out[i]));
          pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
          peak = Math.max(peak, Math.abs(v));
        }
        const now = performance.now();
        if (now - lastUi > 100) { lastUi = now; const lv = peak; peak = 0; setDbg((d) => ({ ...d, level: lv })); }
        let bin = "";
        const u8 = new Uint8Array(pcm.buffer);
        for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
        send({ type: "input_audio_buffer.append", audio: btoa(bin) });
      };
      const src = ac.createMediaStreamSource(stream);
      const mute = ac.createGain();
      mute.gain.value = 0; // keep the graph pulling without echoing the mic
      src.connect(node).connect(mute).connect(ac.destination);

      const say = (role: "user" | "assistant", content: string) => { if (content.trim()) setMsgs((m) => [...m, { role, content }]); };
      const stopPlayback = () => { playing.forEach((n) => { try { n.stop(); } catch {} }); playing.clear(); playAt = 0; };
      const playChunk = (b64: string) => {
        const bin = atob(b64);
        const f = new Float32Array(bin.length >> 1);
        for (let i = 0; i < f.length; i++) { const v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); f[i] = (v > 32767 ? v - 65536 : v) / 32768; }
        const buf = ac.createBuffer(1, f.length, 24000);
        buf.copyToChannel(f, 0);
        const bs = ac.createBufferSource();
        bs.buffer = buf;
        bs.connect(ac.destination);
        playAt = Math.max(playAt, ac.currentTime + 0.05);
        bs.start(playAt);
        playAt += buf.duration;
        playing.add(bs);
        bs.onended = () => playing.delete(bs);
      };

      sock.onopen = () => setVoice("live");
      sock.onclose = (ev) => { if (rtc.current) { if (!ev.wasClean || ev.code !== 1000) setError(`Voice connection closed (${ev.code}) ${ev.reason}`); stopVoice(); } };
      sock.onerror = () => setError("Voice connection error (could not reach api.openai.com - check network/extensions)");
      sock.onmessage = (ev) => {
        const e = JSON.parse(ev.data);
        if (!e.type.endsWith(".delta")) console.debug("[realtime]", e.type, e);
        setDbg((d) => ({ ...d, events: d.events + 1, last: e.type }));
        if (e.type === "response.output_audio.delta") playChunk(e.delta);
        else if (e.type === "input_audio_buffer.speech_started") { stopPlayback(); setLive({ user: "…", assistant: "" }); }
        else if (e.type === "conversation.item.input_audio_transcription.delta") setLive((l) => ({ ...l, user: (l.user === "…" ? "" : l.user) + (e.delta ?? "") }));
        else if (e.type === "response.output_audio_transcript.delta") setLive((l) => ({ ...l, assistant: l.assistant + (e.delta ?? "") }));
        else if (e.type === "conversation.item.input_audio_transcription.completed") { say("user", e.transcript ?? ""); setLive((l) => ({ ...l, user: "" })); }
        else if (e.type === "conversation.item.input_audio_transcription.failed") setError(`Transcription failed: ${e.error?.message ?? "unknown"}`);
        else if (e.type === "response.output_audio_transcript.done") { say("assistant", e.transcript ?? ""); setLive((l) => ({ ...l, assistant: "" })); }
        else if (e.type === "response.done") {
          const calls = (e.response?.output ?? []).filter((o: { type: string }) => o.type === "function_call");
          for (const c of calls) {
            let out: string;
            try {
              out = c.name === "get_scene_context" ? JSON.stringify(ctxRef.current) : onToolRef.current(c.name, JSON.parse(c.arguments || "{}"));
            } catch (err) {
              out = `error: ${err instanceof Error ? err.message : String(err)}`;
            }
            send({ type: "conversation.item.create", item: { type: "function_call_output", call_id: c.call_id, output: out } });
          }
          if (calls.length) send({ type: "response.create" });
        } else if (e.type === "error") setError(e.error?.message ?? "Voice error");
      };
    } catch (e) {
      setError(e instanceof Error ? `${e.name}: ${e.message}` : String(e));
      cleanup();
      rtc.current = null;
      setVoice("off");
    }
  }
  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [msgs, busy, open]);

  async function send(text: string) {
    const t = text.trim();
    if (!t || busy) return;
    setInput("");
    setError("");
    setBusy(true);
    let conv: Msg[] = [...msgs, { role: "user", content: t }];
    setMsgs(conv);
    try {
      for (let round = 0; round < 5; round++) {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: conv, context: ctxRef.current }),
        });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error ?? `Request failed (${res.status})`);
        const m = json.message as { content: string | null; tool_calls?: ToolCall[] };
        const asst: Msg = { role: "assistant", content: m.content, ...(m.tool_calls?.length ? { tool_calls: m.tool_calls } : {}) };
        conv = [...conv, asst];
        if (!m.tool_calls?.length) break;
        for (const tc of m.tool_calls) {
          let out: string;
          try {
            out = onTool(tc.function.name, JSON.parse(tc.function.arguments || "{}"));
          } catch (e) {
            out = `error: ${e instanceof Error ? e.message : String(e)}`;
          }
          conv = [...conv, { role: "tool", tool_call_id: tc.id, content: out }];
        }
        setMsgs(conv);
      }
      setMsgs(conv);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMsgs(conv);
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button onClick={() => setOpen(true)}
        className="btn btn-primary absolute bottom-4 left-4 h-9 rounded-full px-4">
        Ask about the scene
      </button>
    );
  }
  const shown = msgs.filter((m) => m.role === "user" || (m.role === "assistant" && m.content));
  const tools = msgs.filter((m) => m.role === "assistant" && m.tool_calls?.length).length;
  return (
    <div className="panel absolute bottom-4 left-4 flex h-[360px] w-[340px] max-w-[calc(100%-2rem)] flex-col p-0">
      <div className="panel-head m-0 px-4">
        <span className="t-section">Ask about the scene</span>
        <span className="flex items-center gap-2">
          <button onClick={voice === "off" ? startVoice : stopVoice}
            className={`btn btn-sm ${voice === "live" ? "btn-danger" : ""}`}>
            {voice === "off" ? "Voice" : voice === "connecting" ? "Connecting…" : "End voice"}
          </button>
          {tools > 0 && <span className="t-small faint">{tools} tool call round{tools > 1 ? "s" : ""}</span>}
          <button className="btn btn-icon" onClick={() => setOpen(false)} aria-label="Collapse">–</button>
        </span>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto px-4 py-3">
        {shown.length === 0 && (
          <div className="space-y-1.5">
            <p className="muted">Ask about the stress-test results or a muscle. Try:</p>
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => send(s)} className="btn h-auto w-full justify-start py-1.5 text-left">{s}</button>
            ))}
          </div>
        )}
        {shown.map((m, i) => (
          <div key={i} className={`bubble ${m.role === "user" ? "bubble-user" : "bubble-bot"}`}>
            <Markdown text={(m as { content: string }).content} />
          </div>
        ))}
        {voice !== "off" && (
          <div className="space-y-0.5 text-[10px] text-zinc-500">
            <div className="h-1.5 w-full overflow-hidden rounded bg-white/10"><div className="h-full bg-emerald-400" style={{ width: `${Math.min(dbg.level * 300, 100)}%` }} /></div>
            <div>mic level · {dbg.events} events · last: {dbg.last || "none"}</div>
          </div>
        )}
        {voice === "live" && !live.user && !live.assistant && <div className="text-emerald-300">🎤 Listening… speak now</div>}
        {live.user && <div className="ml-8 rounded bg-sky-500/20 px-2 py-1 italic">{live.user}</div>}
        {live.assistant && <div className="mr-4 rounded bg-white/5 px-2 py-1 italic">{live.assistant}</div>}
        {busy && <div className="faint">Thinking…</div>}
        {error && <div className="bubble text-(--danger) bg-[rgba(251,113,133,0.15)]">{error}</div>}
        <div ref={endRef} />
      </div>
      <form className="flex gap-1.5 border-t border-(--line) p-3" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question (any language)…"
          className="field min-w-0 flex-1" />
        <button disabled={busy || !input.trim()} className="btn btn-primary">Send</button>
      </form>
    </div>
  );
}
