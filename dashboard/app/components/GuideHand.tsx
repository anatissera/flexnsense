"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { gramsForceToNewtons, newtonsToGramsForce } from "../lib/units";
import type { ExerciseMode } from "../lib/session/modes";
import type { HandFrame } from "../lib/handFrame";
import { blockFromGrams, type Block } from "../lib/session/protocol";
import type { RepStep } from "../lib/session/reps";
import { DEMO_PASSIVE_TARGET_M, RepSimulation } from "../lib/session/simulation";
import { Bar } from "./charts";

// Index joint targets (rad) when flexed: MCP, PIP, DIP.
const FLEX = [0.5, 0.55, 0.35];
const GREY = new THREE.Color(0xc9ccd2);
// Blue asks the patient to move (a demand). Amber shows what the arm is about to do to them (a
// preview). They must look different: a patient who reads a preview as a demand pushes against
// the arm, and a force cap cannot tell that apart from a stiff joint.
const BLUE = new THREE.Color(0x3b82f6);
const AMBER = new THREE.Color(0xf59e0b);
const MODEL_URL = "/models/hand/right.glb"; // WebXR generic hand, MIT (immersive-web/webxr-input-profiles)
// In this glb the finger bones are siblings under one Armature (not a parent/child chain),
// so the chain is posed by hand. Fingers point -Y, the palm faces -X, so flexion is a
// rotation about world -Z
// (the model is turned palm-up in the scene, which keeps that axis).
const INDEX_BONES = [
  "index-finger-phalanx-proximal",
  "index-finger-phalanx-intermediate",
  "index-finger-phalanx-distal",
  "index-finger-tip",
];
const FLEX_AXIS = new THREE.Vector3(0, 0, -1);

/** 0 = no tint, 1 = demand (the patient moves), 2 = preview (the arm moves). */
type Tint = 0 | 1 | 2;
const RUNNING = new Set(["armed", "cueing", "ramp", "work", "hold", "reset"]);
/**
 * How far the guide hand flexes in each phase, as a fraction of the full flexion target, so the
 * hand does what the cue says rather than following the simulated finger (which only starts
 * moving once the force reaches the cap, well after "Start flexing" is spoken). A phase that is
 * missing means relaxed (0), and null means stay where it is ("Hold it there").
 *
 * Active: flex gently while told to flex, hold, then relax while the arm brings the finger back.
 * Passive: stay relaxed while the arm moves the finger, then flex gently when told to flex back.
 */
const GENTLE_FLEX = 0.55;
const GUIDE_FLEX: Record<ExerciseMode, Partial<Record<string, number | null>>> = {
  active: { cueing: GENTLE_FLEX, ramp: GENTLE_FLEX, work: GENTLE_FLEX, hold: null },
  passive: { hold: null, reset: GENTLE_FLEX },
};
/** Time constant of the hand's approach to its target. Long, so the movement is slow and gentle. */
const FLEX_TAU_S = 0.7;
/** Phases too short to be worth cutting off the cue before them with a second sentence. */
const SILENT_ON_VOICE = new Set(["ramp", "work"]);

function speak(text: string) {
  try {
    const synth = window.speechSynthesis;
    if (!synth) return;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    synth.speak(u);
  } catch {
    /* speech is optional */
  }
}

type View = { step: RepStep; forceN: number | null; travelMm: number; capN: number; reps: number };

/**
 * The rep machine, shown as a hand, a cue and a voice.
 *
 * This used to be a "Flex guide" that played a fixed timeline (5 reps, 3 s flex, 3 s release) on a
 * `performance.now()` clock. It is now driven by the per-repetition machine (`session/reps.py`,
 * ported to `lib/session/reps.ts`) against a SIMULATED patient and arm, so the cue, the hand and
 * the voice all come from one state: a rep is only counted when the force reached the cap, and a
 * lost force reading stops a pulling arm. The hand is the same WebXR hand, and the voice speaks
 * the machine's instruction table (`lib/session/instructions.ts`) instead of "Flex" and "Release".
 *
 * Nothing here is measured and nothing here can move an arm. The machine that decides when a real
 * arm pulls a real finger is the Python one, behind the safety gate.
 */
export default function GuideHand({ subscribe, trackerOn, onTrackerChange }: {
  /** The tracker's HandFrame stream. A frame arrives only when a hand is found. */
  subscribe: (fn: (frame: HandFrame) => void) => () => void;
  /** Whether the webcam hand tracker is running. */
  trackerOn: boolean;
  /** Ask for the webcam hand tracker to be switched on or off. */
  onTrackerChange: (on: boolean) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const flexRef = useRef(0);
  const colorRef = useRef<Tint>(0);
  const simRef = useRef<RepSimulation | null>(null);
  const [playing, setPlaying] = useState(false);
  // Start opens the webcam and waits here until the tracker reports a hand, then the sequence begins.
  const [waiting, setWaiting] = useState(false);
  const waitingRef = useRef(false);
  const pendingBlock = useRef<Block | null>(null);
  const sawTrackerOn = useRef(false);
  const [voice, setVoice] = useState(true);
  const voiceRef = useRef(true);
  const [mode, setMode] = useState<ExerciseMode>("active");
  const [reps, setReps] = useState(3);
  const [capGf, setCapGf] = useState(200);
  const [dropout, setDropout] = useState(false);
  const dropoutRef = useRef(false);
  const [view, setView] = useState<View | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { voiceRef.current = voice; }, [voice]);
  useEffect(() => { dropoutRef.current = dropout; }, [dropout]);
  const said = useRef("");
  const say = useCallback((key: string, text: string) => {
    if (said.current === key) return;
    said.current = key;
    if (voiceRef.current) speak(text);
  }, []);
  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  // Scene setup, once.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const w = mount.clientWidth || 224;
    const h = 150;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, w / h, 0.01, 10);
    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(0.5, 1, 0.3);
    scene.add(dir);

    let disposed = false;
    let raf = 0;
    let bones: THREE.Bone[] = [];
    let restPos: THREE.Vector3[] = [];
    let restQuat: THREE.Quaternion[] = [];
    const parentInv = new THREE.Matrix4();
    const parentQInv = new THREE.Quaternion();
    let controls: OrbitControls | null = null;
    let colorAttr: THREE.BufferAttribute | null = null;
    let weights: Float32Array | null = null;
    let lastTint: Tint | null = null;
    const q = new THREE.Quaternion();
    const rot = new THREE.Quaternion();
    const p = new THREE.Vector3();
    const d = new THREE.Vector3();

    new GLTFLoader().load(MODEL_URL, (gltf) => {
      if (disposed) return;
      const root = gltf.scene;
      // Supination: turn the hand so the palm faces up (+Y) and the fingers point to -X.
      root.rotation.z = -Math.PI / 2;
      scene.add(root);
      root.updateMatrixWorld(true);
      let skinned: THREE.SkinnedMesh | null = null;
      root.traverse((o) => {
        if ((o as THREE.SkinnedMesh).isSkinnedMesh) skinned = o as THREE.SkinnedMesh;
      });
      const mesh = skinned as THREE.SkinnedMesh | null;
      if (!mesh) return;

      // Per-vertex weight on the index finger bones, so only that finger is tinted.
      const names = mesh.skeleton.bones.map((b) => b.name);
      const isIdx = names.map((n) => n.startsWith("index-finger"));
      const geo = mesh.geometry;
      const si = geo.getAttribute("skinIndex");
      const sw = geo.getAttribute("skinWeight");
      weights = new Float32Array(si.count);
      for (let v = 0; v < si.count; v++) {
        let t = 0;
        for (let k = 0; k < 4; k++) if (isIdx[si.getComponent(v, k)]) t += sw.getComponent(v, k);
        weights[v] = Math.min(1, t);
      }
      colorAttr = new THREE.BufferAttribute(new Float32Array(si.count * 3), 3);
      geo.setAttribute("color", colorAttr);
      mesh.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0 });

      bones = INDEX_BONES.map((n) => root.getObjectByName(n) as THREE.Bone);
      restPos = bones.map((b) => b.getWorldPosition(new THREE.Vector3()));
      restQuat = bones.map((b) => b.getWorldQuaternion(new THREE.Quaternion()));
      const parent = bones[0].parent as THREE.Object3D;
      parentInv.copy(parent.matrixWorld).invert();
      parent.getWorldQuaternion(parentQInv).invert();

      // Frame the camera on the hand from a three-quarter side view.
      const box = new THREE.Box3().setFromObject(root);
      const c = box.getCenter(new THREE.Vector3());
      const r = box.getSize(new THREE.Vector3()).length();
      camera.position.set(c.x + r * 0.15, c.y + r * 0.55, c.z + r * 1.0);
      camera.lookAt(c);
      controls = new OrbitControls(camera, renderer.domElement);
      controls.target.copy(c);
      controls.enablePan = false;
      controls.enableZoom = false;
      controls.enableDamping = true;
      controls.update();
      renderer.domElement.style.cursor = "grab";
      renderer.domElement.style.touchAction = "none";
    });

    const tick = () => {
      const f = flexRef.current;
      if (bones.length) {
        // Forward kinematics: each joint pivots about its own rest position, carrying
        // everything distal to it. cum = total flexion accumulated up to that joint.
        let cum = 0;
        p.copy(restPos[0]);
        bones.forEach((b, i) => {
          if (i > 0) {
            rot.setFromAxisAngle(FLEX_AXIS, cum);
            d.copy(restPos[i]).sub(restPos[i - 1]).applyQuaternion(rot);
            p.add(d);
          }
          cum += (FLEX[i] ?? 0) * f;
          q.setFromAxisAngle(FLEX_AXIS, cum).multiply(restQuat[i]);
          b.position.copy(p).applyMatrix4(parentInv);
          b.quaternion.copy(parentQInv).multiply(q);
        });
      }
      controls?.update();
      const tint = colorRef.current;
      if (colorAttr && weights && tint !== lastTint) {
        lastTint = tint;
        const target = tint === 2 ? AMBER : BLUE;
        const tmp = new THREE.Color();
        for (let v = 0; v < weights.length; v++) {
          tmp.copy(GREY).lerp(target, tint ? weights[v] : 0);
          colorAttr.setXYZ(v, tmp.r, tmp.g, tmp.b);
        }
        colorAttr.needsUpdate = true;
      }
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      controls?.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        if (m.material) (m.material as THREE.Material).dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  /** Switch the camera off again. Start is the only control that turns it on, so Stop and the end of a set turn it off. */
  const releaseCamera = useCallback(() => { onTrackerChange(false); }, [onTrackerChange]);

  const start = () => {
    setError(null);
    try {
      // Built now so a bad prescription is reported before the camera is touched.
      pendingBlock.current = blockFromGrams({
        finger: "index", mode, reps, capGf, holdS: 1.0,
        romTargetM: mode === "passive" ? DEMO_PASSIVE_TARGET_M : null,
      });
      new RepSimulation(pendingBlock.current, 0); // validates the block against the machine's own checks
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return;
    }
    said.current = "";
    window.speechSynthesis?.cancel();
    setView(null);
    flexRef.current = 0;
    colorRef.current = 0;
    waitingRef.current = true;
    setWaiting(true);
    say("waiting", "Show your hand to the camera");
    if (!trackerOn) onTrackerChange(true);
  };

  const stop = () => {
    window.speechSynthesis?.cancel();
    flexRef.current = 0;
    colorRef.current = 0;
    if (waitingRef.current) {
      // Cancelled before a hand was ever seen: there is no set to abandon.
      waitingRef.current = false;
      setWaiting(false);
      releaseCamera();
      return;
    }
    const sim = simRef.current;
    if (!sim) return;
    const step = sim.machine.abandon("stopped by the user");
    setView((v) => (v ? { ...v, step } : v));
    setPlaying(false);
    releaseCamera();
  };

  // The first tracked hand starts the sequence. The tracker only reports frames in which it found a hand.
  useEffect(() => {
    return subscribe(() => {
      if (!waitingRef.current || !pendingBlock.current) return;
      waitingRef.current = false;
      setWaiting(false);
      simRef.current = new RepSimulation(pendingBlock.current, performance.now() / 1000);
      setPlaying(true);
    });
  }, [subscribe]);

  // If the tracker is switched off (or fails to start) while still waiting for a hand, stop waiting.
  useEffect(() => {
    if (trackerOn) {
      sawTrackerOn.current = true;
      return;
    }
    if (waitingRef.current && sawTrackerOn.current) {
      waitingRef.current = false;
      setWaiting(false);
      setError("Hand tracking stopped before a hand was seen. Check the camera and try again.");
    }
    sawTrackerOn.current = false;
  }, [trackerOn]);

  // The rep machine loop while playing: one decision per frame, feeding the hand, the voice and the panel.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let lastUi = 0;
    let lastFrame = performance.now();
    let flexTarget = 0;
    const loop = () => {
      const sim = simRef.current;
      if (!sim) return;
      const now = performance.now();
      const r = sim.tick(now / 1000, dropoutRef.current);
      const { step } = r;
      // Ease toward the pose the cue asks for. Null means hold the current target.
      const asked = GUIDE_FLEX[sim.machine.block.mode][step.phase];
      if (step.finished) flexTarget = 0;
      else if (asked !== null) flexTarget = asked ?? 0;
      const dtS = Math.min((now - lastFrame) / 1000, 0.1);
      lastFrame = now;
      flexRef.current += (flexTarget - flexRef.current) * (1 - Math.exp(-dtS / FLEX_TAU_S));
      colorRef.current = RUNNING.has(step.phase) ? (step.driver === "patient" ? 1 : 2) : 0;
      if (!SILENT_ON_VOICE.has(step.phase) && step.phase !== "idle") {
        say(`${step.repsCompleted}-${step.attemptsAbandoned}-${step.phase}`, step.instruction);
      }
      const block = sim.machine.block;
      if (step.finished || now - lastUi > 100) {
        lastUi = now;
        setView({ step, forceN: r.forceN, travelMm: r.travelM * 1000, capN: block.capN, reps: block.reps });
      }
      if (step.finished) {
        setPlaying(false);
        releaseCamera();
        return;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, say, releaseCamera]);

  const step = view?.step ?? null;
  const patientMoves = step !== null && RUNNING.has(step.phase) && step.driver === "patient";
  const stopped = step !== null && (step.phase === "aborted" || step.phase === "set_done");

  return (
    <div id="flex-guide" className="panel absolute bottom-[392px] left-4 w-[340px] max-w-[calc(100%-2rem)] space-y-2 p-3">
      <div className="flex items-center justify-between">
        <span className="t-section">Rep machine</span>
        <span className="chip">
          {waiting ? "Waiting for a hand" : step === null ? "Ready" : stopped ? "Stopped" : `Rep ${step.repIndex}/${view?.reps} · ${step.phase.replace("_", " ")}`}
        </span>
      </div>
      <div ref={mountRef} className="h-[150px] w-full overflow-hidden rounded-md" />
      <div className="flex items-center gap-2">
        {step !== null && !stopped && (
          <span
            className="chip"
            style={{ background: patientMoves ? "var(--warn)" : "var(--surface-raised)", color: patientMoves ? "#0b0d11" : undefined }}
          >
            {patientMoves ? "Your move" : "The arm is moving"}
          </span>
        )}
        <span className="t-body font-semibold">{waiting ? "Show your hand to the camera" : step === null ? "Ready when you are" : step.instruction}</span>
      </div>
      {step !== null && view && (
        <div className="space-y-1">
          <div className="flex justify-between t-small num">
            <span className="muted">Force</span>
            <span>
              {view.forceN === null ? "no reading" : `${view.forceN.toFixed(2)} N (${newtonsToGramsForce(view.forceN).toFixed(0)} gf)`}
              {" "}of cap {view.capN.toFixed(2)} N
            </span>
          </div>
          <Bar value={view.forceN === null ? 0 : view.forceN / view.capN} color={view.forceN === null ? "var(--danger)" : "#4df5e8"} />
          <div className="flex justify-between t-small num">
            <span className="muted">Travel · reps counted</span>
            <span>{view.travelMm.toFixed(0)} mm · {step.repsCompleted} of {view.reps} ({step.attemptsAbandoned} abandoned)</span>
          </div>
          <p className="t-small faint">{step.note}</p>
          {stopped && (
            <p className="t-small" style={{ color: step.phase === "aborted" ? "var(--danger)" : "var(--ink)" }}>
              {step.phase === "aborted"
                ? `Stopped: ${step.note}`
                : `Set finished: ${step.repsCompleted} of ${view.reps} counted, ${step.attemptsAbandoned} abandoned.`}
            </p>
          )}
        </div>
      )}
      {error && <p className="t-small" style={{ color: "var(--danger)" }}>{error}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <select className="btn btn-sm" value={mode} disabled={playing || waiting} onChange={(e) => setMode(e.target.value as ExerciseMode)}>
          <option value="active" className="text-black">Active (you flex)</option>
          <option value="passive" className="text-black">Passive (arm pulls)</option>
        </select>
        <select className="btn btn-sm" value={reps} disabled={playing || waiting} onChange={(e) => setReps(+e.target.value)}>
          {[1, 3, 5].map((n) => <option key={n} value={n} className="text-black">{n} reps</option>)}
        </select>
        <select className="btn btn-sm" value={capGf} disabled={playing || waiting} onChange={(e) => setCapGf(+e.target.value)}>
          {[100, 200, 300].map((g) => <option key={g} value={g} className="text-black">cap {gramsForceToNewtons(g).toFixed(2)} N ({g} gf)</option>)}
        </select>
      </div>
      <div className="flex gap-2">
        {playing || waiting ? (
          <button className="btn btn-danger flex-1" onClick={stop}>
            Stop
          </button>
        ) : (
          <button className="btn btn-primary flex-1" onClick={start}>
            {view ? "Run again" : "Start"}
          </button>
        )}
        <button
          className="btn"
          aria-pressed={voice}
          onClick={() => {
            if (voice) window.speechSynthesis?.cancel();
            setVoice(!voice);
          }}
        >
          {voice ? "Voice on" : "Voice off"}
        </button>
      </div>
      <label className="t-small muted flex items-center gap-1">
        <input type="checkbox" checked={dropout} onChange={(e) => setDropout(e.target.checked)} />
        Simulate a lost force reading (try it while the arm is pulling)
      </label>
      <p className="t-small faint">
        Start opens the webcam, with the model chosen in the right panel, and begins once a hand is detected. The force and arm are simulated (no gauge or arm is attached), so the numbers are not measurements.
      </p>
    </div>
  );
}
