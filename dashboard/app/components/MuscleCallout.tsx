"use client";

import { useEffect, useState, type RefObject } from "react";
import type { MuscleAnchor } from "./MyoViewer";
import { muscleInfo } from "../lib/chatContext";
import type { Study } from "../lib/curlsim/study";
import { FLEXOR_MUSCLES } from "../lib/curlsim/types";

type Anchors = RefObject<{ wanted: Set<string>; pos: Record<string, MuscleAnchor> }>;
const BOX_W = 230;
const BOX_H = 150;

function props(study: Study, week: number, name: string) {
  const info = muscleInfo(name);
  const rows: [string, string][] = [];
  if (info) rows.push(["Role", info.role]);
  for (const s of study.structures.filter((x) => x.session === week)) {
    const m = FLEXOR_MUSCLES[s.finger];
    const isFdp = m.fdp === name;
    const isFds = m.fds === name;
    if (!isFdp && !isFds) continue;
    const est = isFdp ? s.fdp : s.fds;
    const truth = isFdp ? s.trueFdp : s.trueFds;
    const fit = study.structureFits[`${s.finger}:${isFdp ? "fdp" : "fds"}`];
    rows.push(["Est. recovery (wk " + week + ")", est == null ? "–" : `${(est * 100).toFixed(0)}%`]);
    if (truth != null) rows.push(["Sim truth", `${(Math.min(truth, 1.2) * 100).toFixed(0)}%`]);
    if (fit) rows.push(["Fit R∞ · τ", `${(fit.rinf * 100).toFixed(0)}% · ${fit.tau.toFixed(1)} wk`]);
  }
  return { info, rows };
}

export default function MuscleCallout({
  callouts, anchors, study, week, onClose,
}: { callouts: string[]; anchors: Anchors; study: Study; week: number; onClose: (m: string) => void }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!callouts.length) return;
    let raf = 0;
    let last = 0;
    const loop = (t: number) => {
      raf = requestAnimationFrame(loop);
      if (t - last > 33) { last = t; setTick((n) => (n + 1) % 1e9); }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [callouts.length]);

  if (!callouts.length) return null;
  const W = typeof window === "undefined" ? 1200 : window.innerWidth;
  const H = typeof window === "undefined" ? 800 : window.innerHeight;
  const maxX = Math.max(16, W - 486 - BOX_W - 16); // keep clear of the sidebar

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <svg className="absolute inset-0 h-full w-full">
        <defs>
          <marker id="callout-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="#fde047" />
          </marker>
        </defs>
        {callouts.map((name, i) => {
          const a = anchors.current?.pos[name];
          if (!a?.visible) return null;
          const bx = Math.min(Math.max(a.x - BOX_W - 60, 16), maxX);
          const by = Math.min(Math.max(a.y - BOX_H - 30 + i * 36, 16), H - BOX_H - 16);
          const ex = bx + BOX_W; // arrow starts at the box's right edge, middle
          const ey = by + BOX_H / 2;
          return (
            <g key={name}>
              <line x1={ex} y1={ey} x2={a.x} y2={a.y} stroke="#fde047" strokeWidth={1.5} markerEnd="url(#callout-arrow)" />
              <circle cx={a.x} cy={a.y} r={4} fill="none" stroke="#fde047" strokeWidth={1.5} />
            </g>
          );
        })}
      </svg>
      {callouts.map((name, i) => {
        const a = anchors.current?.pos[name];
        if (!a?.visible) return null;
        const bx = Math.min(Math.max(a.x - BOX_W - 60, 16), maxX);
        const by = Math.min(Math.max(a.y - BOX_H - 30 + i * 36, 16), H - BOX_H - 16);
        const { info, rows } = props(study, week, name);
        return (
          <div key={name} className="panel pointer-events-auto absolute t-small !border-(--annotate)/60 !p-3"
            style={{ left: bx, top: by, width: BOX_W, minHeight: BOX_H }}>
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="t-section text-(--annotate)">{name}</div>
                <div className="muted">{info?.label}</div>
              </div>
              <button className="btn btn-icon" onClick={() => onClose(name)} aria-label="Close">✕</button>
            </div>
            <div className="mt-2">
              <div className="kv"><span>Activation now</span><span className="num">{(a.act * 100).toFixed(0)}%</span></div>
              {rows.map(([k, v]) => (
                <div key={k} className={k === "Role" ? "muted py-1" : "kv"}>
                  {k === "Role" ? v : (<><span>{k}</span><span className="num">{v}</span></>)}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
