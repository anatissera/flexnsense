"use client";

import type { ReactNode } from "react";

export const FINGER_COLOR: Record<string, string> = {
  thumb: "#f59e0b",
  index: "#38bdf8",
  middle: "#a78bfa",
  ring: "#34d399",
  little: "#f472b6",
};

export interface Series {
  name: string;
  color: string;
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
  dashed?: boolean;
  dots?: boolean;
  width?: number;
}
export interface Band { x0: number; x1: number; color: string }

const ticks = (lo: number, hi: number, n = 4) => {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
};

export function LineChart({
  series, w = 420, h = 150, xLabel, yLabel, yDomain, xDomain, cursor, bands = [], children,
}: {
  series: Series[]; w?: number; h?: number; xLabel?: string; yLabel?: string;
  yDomain?: [number, number]; xDomain?: [number, number]; cursor?: number; bands?: Band[]; children?: ReactNode;
}) {
  // With a y label, reserve a row above the plot for it. It used to sit on the top tick's row,
  // so a long label ran straight over the top y tick number.
  const m = { l: 38, r: 8, t: yLabel ? 22 : 8, b: 26 };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) for (let i = 0; i < s.xs.length; i++) {
    const x = s.xs[i], y = s.ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  if (xDomain) [x0, x1] = xDomain;
  if (yDomain) [y0, y1] = yDomain;
  if (!Number.isFinite(x0)) return <svg viewBox={`0 0 ${w} ${h}`} className="w-full" />;
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + 1;
  const px = (x: number) => m.l + ((x - x0) / (x1 - x0)) * (w - m.l - m.r);
  const py = (y: number) => h - m.b - ((y - y0) / (y1 - y0)) * (h - m.t - m.b);
  const path = (s: Series) => {
    let d = "";
    let pen = false;
    // decimate long traces
    const step = Math.max(1, Math.floor(s.xs.length / 400));
    for (let i = 0; i < s.xs.length; i += step) {
      if (!Number.isFinite(s.ys[i])) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${px(s.xs[i]).toFixed(1)},${py(s.ys[i]).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full select-none" role="img">
      {bands.map((b, i) => (
        <rect key={i} x={px(b.x0)} y={m.t} width={Math.max(px(b.x1) - px(b.x0), 0)} height={h - m.t - m.b} fill={b.color} />
      ))}
      {ticks(y0, y1).map((v) => (
        <g key={`y${v}`}>
          <line x1={m.l} x2={w - m.r} y1={py(v)} y2={py(v)} stroke="currentColor" opacity={0.1} />
          <text x={m.l - 4} y={py(v) + 3} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.6}>{v}</text>
        </g>
      ))}
      {ticks(x0, x1, 5).map((v) => (
        <text key={`x${v}`} x={px(v)} y={h - m.b + 11} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.6}>{v}</text>
      ))}
      {xLabel && <text x={(m.l + w - m.r) / 2} y={h - 2} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.7}>{xLabel}</text>}
      {yLabel && <text x={4} y={11} fontSize={9} fill="currentColor" opacity={0.7}>{yLabel}</text>}
      {series.map((s) => (
        <g key={s.name}>
          <path d={path(s)} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.5} strokeDasharray={s.dashed ? "4 3" : undefined} />
          {s.dots && Array.from(s.xs).map((x, i) => Number.isFinite(s.ys[i]) && (
            <circle key={i} cx={px(x)} cy={py(s.ys[i])} r={2.8} fill={s.color} />
          ))}
        </g>
      ))}
      {cursor !== undefined && cursor >= x0 && cursor <= x1 && (
        <line x1={px(cursor)} x2={px(cursor)} y1={m.t} y2={h - m.b} stroke="#fff" opacity={0.7} />
      )}
      {children}
    </svg>
  );
}

export function Legend({ items }: { items: { name: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 t-small muted">
      {items.map((i) => (
        <span key={i.name} className="inline-flex items-center gap-1">
          <span className="inline-block h-0.5 w-4" style={{ background: i.color, opacity: i.dashed ? 0.6 : 1 }} />
          {i.name}
        </span>
      ))}
    </div>
  );
}

/** Horizontal bar 0..120 % with a 100 % reference tick. */
export function Bar({ value, color }: { value: number; color: string }) {
  const v = Number.isFinite(value) ? Math.min(Math.max(value, 0), 1.2) : 0;
  return (
    <div className="relative h-2 w-full rounded-full bg-(--surface-raised)">
      <div className="h-full rounded-full" style={{ width: `${(v / 1.2) * 100}%`, background: color }} />
      <div className="absolute inset-y-0 w-px bg-(--ink-muted)" style={{ left: `${(1 / 1.2) * 100}%` }} />
    </div>
  );
}
