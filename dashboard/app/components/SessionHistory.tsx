"use client";

import { useState } from "react";
import { reportPage } from "../lib/markdownHtml";
import { adherence, type DemoPatient, type PastSession } from "../lib/demo/patients";
import { pastSessionMd } from "../lib/demo/pastReport";
import { LineChart, type Series } from "./charts";
import DemoBadge from "./DemoBadge";

/**
 * What this patient has actually done, session by session.
 *
 * The column that matters is counted against prescribed, because a repetition only counts when
 * the force reached the cap. That is the number a clinician can act on: a week of short
 * repetitions means the finger, the cap or the plan, and it is visible here before anybody
 * computes a recovery index.
 *
 * Time lost to a missing force reading is shown in its own column, never folded into the others.
 * It is a fact about the camera's view of the scale, and reading it as a worse week would be
 * reading an instrument problem as a patient problem.
 */
export default function SessionHistory({ patient }: { patient: DemoPatient }) {
  const [open, setOpen] = useState<string | null>(null);
  const history = patient.history;
  const adh = adherence(history);

  const openReport = (s: PastSession) => {
    const html = reportPage(`Session report ${patient.id} ${s.date}`, pastSessionMd(patient, s));
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    window.open(url, "_blank");
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };

  const series: Series[] = [
    {
      name: "travel",
      color: "var(--phase-flexion)",
      xs: history.map((s) => s.week),
      ys: history.map((s) => s.romMm),
      dots: true,
    },
  ];

  return (
    <section id="sec-history" className="section space-y-2 border-t border-(--line) pt-3">
      <div className="flex items-center gap-2">
        <h3 className="t-section">Past sessions</h3>
        <DemoBadge kind="simulated" label="Example history" />
      </div>
      <LineChart h={110} yLabel="travel (mm)" xLabel="week" series={series} />
      <table className="tbl">
        <thead>
          <tr><th>Date</th><th>Wk</th><th>Counted</th><th>Cap</th><th>Travel</th><th>No reading</th><th /></tr>
        </thead>
        <tbody>
          {history.map((s) => (
            <tr key={s.date}>
              <td>{s.date}</td>
              <td>{s.week}</td>
              <td className={s.repsAbandoned > 0 ? "text-(--warn)" : undefined}>
                {s.repsCounted}/{s.repsPrescribed}
              </td>
              <td>{s.capGf} gf</td>
              <td>{s.romMm.toFixed(1)} mm</td>
              <td className={s.dropoutS > 5 ? "text-(--danger)" : "muted"}>
                {s.dropoutS === 0 ? "–" : `${s.dropoutS.toFixed(1)} s`}
              </td>
              <td className="text-right">
                <button className="btn btn-sm" onClick={() => setOpen(open === s.date ? null : s.date)}>
                  {open === s.date ? "Hide" : "Note"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {open && (
        <div className="rounded-md border border-(--line) p-2.5 t-small">
          {(() => {
            const s = history.find((h) => h.date === open);
            if (!s) return null;
            return (
              <div className="space-y-2">
                <p>{s.note}</p>
                <div className="flex flex-wrap gap-x-4 gap-y-1 muted">
                  <span>Peak {s.peakGf} gf of {s.capGf} gf cap</span>
                  <span>{s.repsAbandoned} abandoned</span>
                  <span>Recovery index {s.recoveryIndex === null ? "not determined" : `${(s.recoveryIndex * 100).toFixed(0)}%`}</span>
                </div>
                <button className="btn btn-sm" onClick={() => openReport(s)}>Open the full report</button>
              </div>
            );
          })()}
        </div>
      )}
      <p className="t-small faint">
        {history.length} sessions. Adherence {adh === null ? "not determined" : `${(adh * 100).toFixed(0)}%`} of
        prescribed repetitions counted. A repetition counts only when the force reached the cap, so a short
        one is recorded as abandoned rather than quietly passing.
      </p>
    </section>
  );
}
