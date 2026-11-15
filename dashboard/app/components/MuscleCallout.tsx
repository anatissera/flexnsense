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

const GAP = 8;
const MARGIN = 16;
type Box = { name: string; bx: number; by: number; h: number };

/**
 * Where each box goes. Every box starts at its preferred spot beside its anchor; then, top to
 * bottom, any box that would cover one already placed is pushed below it, and if that runs off the
 * bottom edge the stack is pushed back up. Boxes that do not share horizontal space are left alone.
 */
function layoutBoxes(want: Box[], H: number): Record<string, { bx: number; by: number }> {
  const sorted = [...want].sort((a, b) => a.by - b.by);
  const xOverlap = (a: Box, b: Box) => a.bx < b.bx + BOX_W + GAP && b.bx < a.bx + BOX_W + GAP;
  const placed: Box[] = [];
  for (const box of sorted) {
    const b = { ...box };
    // Moving down can create a new overlap with an earlier box, so repeat until nothing overlaps.
    for (let moved = true, guard = 0; moved && guard < 50; guard++) {
      moved = false;
      for (const p of placed) {
        if (xOverlap(b, p) && b.by < p.by + p.h + GAP && p.by < b.by + b.h + GAP) { b.by = p.by + p.h + GAP; moved = true; }
      }
    }
    placed.push(b);
  }
  // Off the bottom: walk back up, lifting each box clear of the one below it.
  for (let i = placed.length - 1; i >= 0; i--) {
    const b = placed[i];
    let limit = H - b.h - MARGIN;
    for (let j = i + 1; j < placed.length; j++) if (xOverlap(b, placed[j])) limit = Math.min(limit, placed[j].by - b.h - GAP);
    b.by = Math.max(Math.min(b.by, limit), MARGIN);
  }
  return Object.fromEntries(placed.map((b) => [b.name, { bx: b.bx, by: b.by }]));
}

export default function MuscleCallout({
  callouts, anchors, study, week, onClose,
}: { callouts: string[]; anchors: Anchors; study: Study; week: number; onClose: (m: string) => void }) {
  const [, setTick] = useState(0);
  // Measured box heights (a long role text wraps), read back on the next frame's layout.
  const [heights] = useState(() => new Map<string, number>());
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

  const want: Box[] = [];
  callouts.forEach((name, i) => {
    const a = anchors.current?.pos[name];
    if (!a?.visible) return;
    const h = Math.max(BOX_H, heights.get(name) ?? 0);
    want.push({
      name, h,
      bx: Math.min(Math.max(a.x - BOX_W - 60, MARGIN), maxX),
      by: Math.min(Math.max(a.y - BOX_H - 30 + i * 36, MARGIN), H - h - MARGIN),
    });
  });
  const spot = layoutBoxes(want, H);

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <svg className="absolute inset-0 h-full w-full">
        <defs>
          <marker id="callout-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="#fde047" />
          </marker>
        </defs>
        {callouts.map((name) => {
          const a = anchors.current?.pos[name];
          const s = spot[name];
          if (!a?.visible || !s) return null;
          const ex = s.bx + BOX_W; // arrow starts at the box's right edge, middle
          const ey = s.by + Math.max(BOX_H, heights.get(name) ?? 0) / 2;
          return (
            <g key={name}>
              <line x1={ex} y1={ey} x2={a.x} y2={a.y} stroke="#fde047" strokeWidth={1.5} markerEnd="url(#callout-arrow)" />
              <circle cx={a.x} cy={a.y} r={4} fill="none" stroke="#fde047" strokeWidth={1.5} />
            </g>
          );
        })}
      </svg>
      {callouts.map((name) => {
        const a = anchors.current?.pos[name];
        const s = spot[name];
        if (!a?.visible || !s) return null;
        const { info, rows } = props(study, week, name);
        return (
          <div key={name} className="panel pointer-events-auto absolute t-small !border-(--annotate)/60 !p-3"
            ref={(el) => { if (el) heights.set(name, el.offsetHeight); else heights.delete(name); }}
            style={{ left: s.bx, top: s.by, width: BOX_W, minHeight: BOX_H }}>
            <div className="flex items-start justify-between gap-2">
              <div className="t-section text-(--annotate)">{info?.label ?? name}</div>
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
