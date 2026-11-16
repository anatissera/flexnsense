"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { gramsForceToNewtons, newtonsToGramsForce } from "../lib/units";
import { Bar } from "./charts";
import DemoBadge from "./DemoBadge";
import { Hint } from "./Hints";
import type { GaugeSample } from "./GaugeForce";

/**
 * The force panel when there is no gauge: a simulation, saying so, in the gauge's own place.
 *
 * At the bench this column is a camera pointed at a PESOLA spring scale, and the number under it
 * is the only force measurement in the whole system. With no bench there are two honest choices,
 * show the panel failing to connect or show a simulation that is labelled: the first teaches a
 * visitor nothing about the product, so this is the second.
 *
 * It has two sources and never both at once. While a set is running the rep machine owns the
 * reading, because that is the force its own decisions are being made on and two different force
 * numbers on one screen would be a lie about which one matters. The rest of the time the slider
 * owns it, so a visitor can push past the cap by hand and watch what the system does about it.
 *
 * It writes the same `GaugeSample` shape the real reader writes, into the same ref, so nothing
 * downstream knows the difference. What downstream must not do is call it a measurement: see the
 * `forceSource` wording in SimApp.
 */

/** The ceiling of the real instrument, so the slider cannot pretend to a force the PESOLA cannot read. */
const MAX_GF = 600;

export default function SimulatedGauge({ forceRef, onActive, driven }: {
  /** The same ref the gauge reader writes, written here with a simulated sample. */
  forceRef: RefObject<GaugeSample | null>;
  /** Called when force sensing is switched on or off, exactly as the real panel calls it. */
  onActive?: (on: boolean) => void;
  /**
   * The rep machine's current force, or null when no set is running. A ref rather than a prop
   * value because it changes every frame: through state it would re-render this panel sixty
   * times a second to move one number.
   */
  driven: RefObject<number | null>;
}) {
  const [on, setOn] = useState(false);
  const [handGf, setHandGf] = useState(0);
  const [shown, setShown] = useState<number | null>(null);
  const [byMachine, setByMachine] = useState(false);

  const onActiveRef = useRef(onActive);
  useEffect(() => { onActiveRef.current = onActive; });
  useEffect(() => {
    onActiveRef.current?.(on);
    return () => { if (on) onActiveRef.current?.(false); };
  }, [on]);

  // One writer, on a frame clock, so the sample's age means the same thing it does with a real
  // reader: a consumer calling freshGauge() sees it go stale the moment this stops publishing.
  useEffect(() => {
    // Switching off is handled by the button, which is where the state it clears belongs.
    if (!on) return;
    let raf = 0;
    let lastUi = 0;
    const tick = (now: number) => {
      const fromMachine = driven.current;
      const n = fromMachine !== null ? fromMachine : gramsForceToNewtons(handGf);
      forceRef.current = { n, g: newtonsToGramsForce(n), stable: true, at: performance.now() };
      // The ref moves every frame; the panel only needs to keep up with an eye.
      if (now - lastUi > 100) {
        lastUi = now;
        setShown(n);
        setByMachine(fromMachine !== null);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      forceRef.current = null;
    };
  }, [on, driven, handGf, forceRef]);

  /** Off means no source, so no reading: the same rule the real panel follows on a dropout. */
  const toggle = () => {
    if (on) {
      setShown(null);
      setByMachine(false);
    }
    setOn(!on);
  };
  return (
    // The same three-row subgrid as the real panel, so the two columns stay aligned.
    <div className="row-span-3 grid grid-rows-subgrid gap-y-2">
      <div className="flex flex-wrap content-start items-center gap-2">
        <DemoBadge kind="simulated" />
        <span className="t-small muted">Force (no gauge attached)</span>
        <button className={on ? "btn btn-danger" : "btn"} onClick={toggle}>
          {on ? "Stop force" : "Start force"}
        </button>
      </div>
      <div>
        {on && (
          <div className="flex aspect-[4/3] w-full flex-col justify-center gap-3 rounded border border-(--line) bg-[rgba(255,255,255,0.03)] p-3">
            <div className="flex items-baseline gap-2">
              <span className="num text-2xl font-semibold">{shown === null ? "–" : shown.toFixed(2)}</span>
              <span className="t-body muted">N</span>
              {shown !== null && (
                <span className="t-small muted">({newtonsToGramsForce(shown).toFixed(0)} gf)</span>
              )}
              <span className="ml-auto t-small faint">{byMachine ? "from the rep machine" : "from the slider"}</span>
            </div>
            <Bar value={shown === null ? 0 : shown / gramsForceToNewtons(MAX_GF)} color="#4df5e8" />
            <label className="t-small block">
              <span className={byMachine ? "faint" : "muted"}>
                Pull on the finger: {handGf} gf ({gramsForceToNewtons(handGf).toFixed(2)} N)
              </span>
              <input
                type="range" min={0} max={MAX_GF} step={5} value={handGf} disabled={byMachine}
                className="w-full"
                onChange={(e) => setHandGf(+e.target.value)}
              />
            </label>
            <p className="t-small faint">
              {byMachine
                ? "A set is running, so this is the force the rep machine is deciding on. The slider comes back when it ends."
                : "Drag it past the clinician's cap and start a set: that is the force the arm has to give way to."}
            </p>
          </div>
        )}
      </div>
      <div className="space-y-2">
        <Hint kind="demo">
          No gauge detected. Start force here anyway: the number is simulated, and while a set runs it
          is the force the rep machine is actually deciding on, so you can watch it reach the cap and
          the arm give way.
        </Hint>
        <p className="t-small muted">
          At the bench this column is a camera watching a PESOLA 600 gf spring scale, reading the red marker
          against the printed scale at about 30 Hz. It is the only force sensor in the system: the servo
          current turned out to be a 1 mA noise floor through a 345:1 gearbox, so there is no second opinion.
        </p>
        <p className="t-small faint">
          Nothing here is measured. With a real gauge, a marker that goes out of view produces no reading at
          all rather than the last one, and the arm releases.
        </p>
      </div>
    </div>
  );
}
