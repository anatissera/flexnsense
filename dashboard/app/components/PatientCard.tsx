"use client";

import { useState } from "react";
import Image from "next/image";
import type { DemoPatient } from "../lib/demo/patients";

/**
 * Who this is, and the controls for moving between them.
 *
 * Previous and Next used to be decoration. They now switch between the two demonstration
 * patients, which is the fastest way to show that the machine is prescribed rather than
 * configured: the same screen, two conditions, two different modes of exercise.
 *
 * Save writes to this browser and says so. A demonstration build has nowhere else to put it, and
 * a Save that silently does nothing is worse than one that tells the truth about its reach.
 */

const SAVED_KEY = "flexnsense.demo.saved.v1";

function Avatar({ src, name }: { src: string | null; name: string }) {
  if (!src) {
    const initials = name.split(" ").map((p) => p[0]).join("").slice(0, 2).toUpperCase();
    return (
      <div
        className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full t-body font-semibold"
        style={{ background: "var(--surface-raised)", color: "var(--ink-muted)" }}
        aria-hidden
      >
        {initials}
      </div>
    );
  }
  return (
    // The file is a 1254 px square, so next/image serves a small copy sized for the circle.
    // Greyscale like the placeholder it replaces: the dashboard keeps colour for data and state.
    <div className="relative h-12 w-12 shrink-0 overflow-hidden rounded-full" style={{ filter: "grayscale(1)" }}>
      <Image
        src={src}
        alt=""
        width={96}
        height={96}
        className="h-full w-full object-cover"
        style={{ transform: "scale(1.45)", transformOrigin: "50% 42%" }}
      />
    </div>
  );
}

export default function PatientCard({ patient, index, count, onPrev, onNext, onExport }: {
  patient: DemoPatient;
  index: number;
  count: number;
  onPrev: () => void;
  onNext: () => void;
  onExport?: () => void;
}) {
  // Tagged with the patient it belongs to, so moving to the next one drops the message without
  // an effect reaching in to clear it.
  const [saved, setSaved] = useState<{ patient: string; text: string } | null>(null);
  const savedHere = saved?.patient === patient.id ? saved.text : null;

  const save = () => {
    const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
    let text: string;
    try {
      window.localStorage.setItem(SAVED_KEY, JSON.stringify({ patient: patient.id, at: stamp }));
      text = `Saved in this browser at ${stamp}. Nothing left this machine.`;
    } catch {
      text = "This browser refused to store it. Nothing left this machine either way.";
    }
    setSaved({ patient: patient.id, text });
  };

  return (
    <section className="space-y-3 border-b border-(--line) pb-4" aria-label="Patient profile">
      <div className="flex items-start gap-3">
        <Avatar src={patient.photo} name={patient.name} />
        <div className="min-w-0">
          <h2 className="t-title">{patient.name}</h2>
          <div className="t-small muted mt-1 space-y-0.5">
            {patient.lines.map((l) => (
              <p key={l}>{l}</p>
            ))}
          </div>
        </div>
      </div>
      <p className="t-small muted">{patient.summary}</p>
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-sm" onClick={onPrev} title="The other demonstration patient">
          Previous patient
        </button>
        <button className="btn btn-sm" onClick={onNext} title="The other demonstration patient">
          Next patient
        </button>
        <span className="t-small faint">{index + 1} of {count}</span>
        <button className="btn btn-sm ml-auto" onClick={save} title="Stores your place in this browser only">
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
      {savedHere && <p className="t-small faint">{savedHere}</p>}
    </section>
  );
}
