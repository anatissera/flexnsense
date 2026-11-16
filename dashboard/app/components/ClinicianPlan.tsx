"use client";

import { useMemo } from "react";
import { newtonsToGramsForce } from "../lib/units";
import { blockCapGf, type Block } from "../lib/session/protocol";
import { planOf, plannedReps, type DemoPatient } from "../lib/demo/patients";
import DemoBadge from "./DemoBadge";

/**
 * The half of the system a clinician owns: what this patient is supposed to do, who said so, and
 * when. It is the panel that makes the machine a prescription rather than a gadget.
 *
 * Every row is a block, and Load puts one into the rep machine. That link is the point: a judge
 * can read a surgeon's instruction and then press Start on exactly that, instead of reading one
 * screen and operating another.
 *
 * The plan is parsed through `protocolFromJSON`, which is the same validation `session/protocol.py`
 * applies on the rig, including the rule that no block's cap may reach the session ceiling. A plan
 * that would be refused there is refused here, and this panel says so rather than rendering it.
 */
export default function ClinicianPlan({ patient, onLoad, loaded }: {
  patient: DemoPatient;
  /** Send one block to the rep machine, with the label to show for where it came from. */
  onLoad: (block: Block, label: string) => void;
  /** The label of the block currently in the rep machine, if it came from here. */
  loaded: string | null;
}) {
  const parsed = useMemo(() => {
    try {
      return { plan: planOf(patient), error: null as string | null };
    } catch (e) {
      return { plan: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [patient]);

  if (parsed.error || !parsed.plan) {
    return (
      <section id="sec-plan" className="section space-y-2 border-t border-(--line) pt-3">
        <h3 className="t-section">Plan from the clinician</h3>
        <p className="t-small" style={{ color: "var(--danger)" }}>
          This plan does not validate, so it is not shown as runnable: {parsed.error}
        </p>
      </section>
    );
  }

  const plan = parsed.plan;
  const ceilingGf = newtonsToGramsForce(plan.ceilingN);
  return (
    <section id="sec-plan" className="section space-y-2 border-t border-(--line) pt-3">
      <div className="flex items-center gap-2">
        <h3 className="t-section">Plan from the clinician</h3>
        <DemoBadge kind="simulated" label="Example plan" />
      </div>
      <div className="t-small muted">
        <p>
          Uploaded {patient.plan.uploadedOn} by {patient.plan.uploadedBy}
        </p>
        <p className="mt-1 italic">&ldquo;{patient.plan.note}&rdquo;</p>
      </div>
      <table className="tbl">
        <thead>
          <tr><th>Finger</th><th>Mode</th><th>Reps</th><th>Cap</th><th>Hold</th><th>Stretch</th><th /></tr>
        </thead>
        <tbody>
          {plan.blocks.map((b, i) => {
            const label = `${b.finger} ${b.mode} ${Math.round(blockCapGf(b))} gf`;
            return (
              <tr key={`${b.finger}-${i}`}>
                <td className="capitalize">{b.finger}</td>
                <td className="uppercase">{b.mode}</td>
                <td>{b.reps}</td>
                <td>{Math.round(blockCapGf(b))} gf</td>
                <td>{b.holdS.toFixed(0)} s</td>
                <td>{b.romTargetM === null ? "–" : `${(b.romTargetM * 1000).toFixed(0)} mm`}</td>
                <td className="text-right">
                  <button
                    className={loaded === label ? "btn btn-sm btn-primary" : "btn btn-sm"}
                    onClick={() => onLoad(b, label)}
                    title="Put this block into the rep machine, bottom left. It does not start it."
                  >
                    {loaded === label ? "Loaded" : "Load"}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="t-small faint">
        {plan.blocks.length} blocks, {plannedReps(plan)} repetitions, session ceiling {ceilingGf.toFixed(0)} gf.
        No block&apos;s cap may reach the ceiling: the controller needs headroom to give way in before the
        safety gate would release the finger altogether. Loading a block fills the rep machine, it does
        not start it.
      </p>
    </section>
  );
}
