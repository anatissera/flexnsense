import type { LiveSnapshot } from "./live";
import { GROUPS, type Study } from "./study";

const pc = (v: number | null | undefined, d = 0) =>
  v == null || !Number.isFinite(v) ? "-" : `${(v * 100).toFixed(d)}%`;
const n = (v: number | null | undefined, d = 1) => (v == null || !Number.isFinite(v) ? "-" : v.toFixed(d));

function band(ri: number | null) {
  if (ri == null || !Number.isFinite(ri)) return "Not determined";
  if (ri >= 0.9) return "Near-normal function";
  if (ri >= 0.7) return "Good progress, mild residual deficit";
  if (ri >= 0.5) return "Partial recovery, moderate deficit";
  return "Marked deficit";
}

/** Clinical-style summary of one completed index-finger curl session. */
export function medicalReportMd(study: Study, week: number, snap: LiveSnapshot): string {
  const reps = snap.reps;
  const ri = snap.recovery_index;
  const fit = study.fits.index;
  const unit = study.structures.find((s) => s.finger === "index" && s.session === week);
  const date = new Date().toISOString().slice(0, 10);
  const med = (k: string, d = 1) => n(snap.med[k], d);

  const L: string[] = [
    `# Hand Therapy Assessment Report`,
    "",
    `- **Patient ID:** ${study.patient.id}`,
    `- **Report date:** ${date}`,
    `- **Post-operative session:** week ${week}`,
    `- **Test:** index finger flexion curl, injured hand compared with contralateral reference`,
    `- **Repetitions analysed:** ${reps.length}`,
    "",
    "## Summary",
    "",
    `**Recovery index: ${pc(ri)} of contralateral. ${band(ri)}.**`,
    "",
  ];

  if (snap.groups) {
    L.push("| Domain | Score vs contralateral |", "|---|---|");
    for (const g of GROUPS) L.push(`| ${g[0].toUpperCase() + g.slice(1)} | ${pc(snap.groups[g])} |`);
    const sorted = [...GROUPS].sort((a, b) => snap.groups![a] - snap.groups![b]);
    L.push("", `Weakest domain: **${sorted[0]}** (${pc(snap.groups[sorted[0]])}). Strongest domain: **${sorted[sorted.length - 1]}** (${pc(snap.groups[sorted[sorted.length - 1]])}).`, "");
  }

  L.push(
    "## Findings",
    "",
    "| Measure | Median | Unit |",
    "|---|---|---|",
    `| Range of motion (fingertip displacement) | ${med("rom")} | mm |`,
    `| Peak fingertip force | ${med("peak_force")} | N |`,
    `| Curl time | ${med("curl_time", 2)} | s |`,
    `| Peak velocity | ${med("peak_velocity", 0)} | mm/s |`,
    `| Rate of force development (200 ms) | ${med("rfd_200", 0)} | N/s |`,
    `| Apparent stiffness | ${med("stiffness", 2)} | N/mm |`,
    `| Hysteresis | ${med("hysteresis", 0)} | % |`,
    `| Movement smoothness (SPARC) | ${med("sparc", 2)} | |`,
    `| Hold steadiness (CV) | ${med("steadiness_cv")} | % |`,
    "",
    `Repeatability: ROM CV ${n(snap.trial.rom_cv)}%, curl-time CV ${n(snap.trial.curl_time_cv)}%, force decay across reps ${n(snap.trial.force_decay)}%.`,
    "",
    "### Per-repetition data",
    "",
    "| Rep | ROM (mm) | Curl time (s) | Peak force (N) | RFD 200 (N/s) | Hysteresis (%) |",
    "|---|---|---|---|---|---|",
    ...reps.map((r) => `| ${r.rep} | ${n(r.rom)} | ${n(r.curl_time, 2)} | ${n(r.peak_force)} | ${n(r.rfd_200, 0)} | ${n(r.hysteresis, 0)} |`),
    "",
    "## Healing course (index finger)",
    "",
  );
  if (fit) {
    L.push(
      `Fitted course R(t) = R_inf (1 - e^(-t/tau)) across sessions: projected plateau ${pc(fit.rinf)}, time constant ${n(fit.tau)} weeks, ` +
        `${Number.isFinite(fit.t90) ? `90% of contralateral expected at about week ${n(fit.t90)}` : "90% of contralateral not expected to be reached"}.`,
    );
  } else L.push("Not enough sessions for a fit.");
  if (unit) {
    L.push("", `Model-attributed flexor tendon unit recovery at week ${week}: FDP ${pc(unit.fdp)}${unit.fds != null ? `, FDS ${pc(unit.fds)}` : ""}.`);
  }

  const notes: string[] = [];
  if (ri != null && ri < 0.7) notes.push("Continue supervised flexor rehabilitation; reassess at the next scheduled session.");
  if (snap.groups && snap.groups.motion < 0.7) notes.push("Reduced range of motion: consider adhesion assessment and tendon-gliding progression.");
  if (snap.groups && snap.groups.strength < 0.7) notes.push("Reduced fingertip force: graded strengthening only as the repair allows.");
  if (Number.isFinite(snap.trial.rom_cv) && snap.trial.rom_cv > 15) notes.push("Variable range between repetitions: check pain, fatigue or technique.");
  if (!notes.length) notes.push("Metrics are near the contralateral reference; continue the current programme and monitor.");
  L.push("", "## Suggested follow-up", "", ...notes.map((x) => `- ${x}`));

  if (study.flags.length) L.push("", "## Data-quality flags", "", ...study.flags.slice(0, 10).map((f) => `- ${f}`));

  L.push(
    "",
    "> All data are simulated. Force-displacement values are apparent whole-system behaviour, not true tendon properties. Only flexor-driven curls were assessed. This report is for research and demonstration and must not be used for clinical decisions.",
  );
  return L.join("\n");
}
