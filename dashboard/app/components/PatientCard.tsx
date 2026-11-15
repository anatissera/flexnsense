// Placeholder patient profile. Static data; the previous, next and save buttons have no action yet.

import Image from "next/image";

const PATIENT = {
  name: "Jordan Woods",
  lines: [
    "58 y, Male | MRN 004-71829",
    "DOB 1967-03-14 | Right-hand dominant",
    "Dx: Flexor tendon repair, zone II (digit 3, R)",
    "Surgery 2026-09-22 | Referred by Dr. A. Meier",
  ],
};

/**
 * The patient's photo, in a 48 px circle cropped in on the face. The file is a 1254 px square, so
 * next/image serves a small copy sized for the circle (96 px for a high-density screen). It stays
 * greyscale like the placeholder it replaces: the dashboard keeps colour for data and state.
 * Decorative, because the name sits right beside it.
 */
function Avatar() {
  return (
    <div className="relative h-12 w-12 shrink-0 overflow-hidden rounded-full" style={{ filter: "grayscale(1)" }}>
      <Image
        src="/profile_photo.png"
        alt=""
        width={96}
        height={96}
        className="h-full w-full object-cover"
        style={{ transform: "scale(1.45)", transformOrigin: "50% 42%" }}
      />
    </div>
  );
}

export default function PatientCard({ onExport }: { onExport?: () => void }) {
  return (
    <section className="space-y-3 border-b border-(--line) pb-4" aria-label="Patient profile">
      <div className="flex items-start gap-3">
        <Avatar />
        <div className="min-w-0">
          <h2 className="t-title">{PATIENT.name}</h2>
          <div className="t-small muted mt-1 space-y-0.5">
            {PATIENT.lines.map((l) => (
              <p key={l}>{l}</p>
            ))}
          </div>
        </div>
      </div>
      <div className="flex gap-2">
        <button className="btn btn-sm" title="Coming soon">
          Previous patient
        </button>
        <button className="btn btn-sm" title="Coming soon">
          Next patient
        </button>
        <button className="btn btn-sm ml-auto" title="Coming soon">
          Save
        </button>
        <button
          className="btn btn-sm"
          disabled={!onExport}
          onClick={onExport}
          title={onExport ? "Open the medical report as a page" : "Complete a curl to enable"}
        >
          Export Medical Report
        </button>
      </div>
    </section>
  );
}
