"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const REPS = 5;
const HOLD_S = 3;
const RELEASE_S = 3;
const REP_S = HOLD_S + RELEASE_S;
const TOTAL_S = REPS * REP_S;
const RAMP_S = 0.6; // time to move between rest and flexed

// Index joint targets (rad) when flexed: MCP, PIP, DIP.
const FLEX = [0.5, 0.55, 0.35];
const GREY = new THREE.Color(0xc9ccd2);
const BLUE = new THREE.Color(0x3b82f6);
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

const ease = (t: number) => t * t * (3 - 2 * t);

/** 0 = rest, 1 = fully flexed, for time `t` seconds into a rep. */
function flexAt(t: number) {
  if (t < HOLD_S) return ease(Math.min(1, t / RAMP_S));
  const r = t - HOLD_S;
  return 1 - ease(Math.min(1, r / RAMP_S));
}

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

export default function GuideHand() {
  const mountRef = useRef<HTMLDivElement>(null);
  const flexRef = useRef(0);
  const colorRef = useRef(false);
  const startRef = useRef<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [voice, setVoice] = useState(true);
  const voiceRef = useRef(true);
  voiceRef.current = voice;
  const said = useRef("");
  const say = useCallback((key: string, text: string) => {
    if (said.current === key) return;
    said.current = key;
    if (voiceRef.current) speak(text);
  }, []);
  useEffect(() => () => window.speechSynthesis?.cancel(), []);
  const [info, setInfo] = useState({ rep: 0, phase: "Ready", progress: 0 });

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
    let lastBlue: boolean | null = null;
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
      const blue = colorRef.current;
      if (colorAttr && weights && blue !== lastBlue) {
        lastBlue = blue;
        const tmp = new THREE.Color();
        for (let v = 0; v < weights.length; v++) {
          tmp.copy(GREY).lerp(BLUE, blue ? weights[v] : 0);
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

  const reset = useCallback(() => {
    startRef.current = null;
    flexRef.current = 0;
    colorRef.current = false;
    said.current = "";
    window.speechSynthesis?.cancel();
    setPlaying(false);
    setInfo({ rep: 0, phase: "Ready", progress: 0 });
  }, []);

  // Timeline loop while playing.
  useEffect(() => {
    if (!playing) return;
    startRef.current = performance.now();
    let raf = 0;
    let lastUi = 0;
    const loop = () => {
      const now = performance.now();
      const el = (now - (startRef.current ?? now)) / 1000;
      if (el >= TOTAL_S) {
        reset();
                return;
      }
      const rep = Math.floor(el / REP_S);
      const t = el - rep * REP_S;
      flexRef.current = flexAt(t);
      const holding = t < HOLD_S;
      colorRef.current = holding;
      if (holding) {
        say(`flex${rep}`, "Flex");
      } else {
        say(`rel${rep}`, "Release");
      }
      if (now - lastUi > 100) {
        lastUi = now;
        setInfo({ rep: rep + 1, phase: holding ? "Flex" : "Release", progress: el / TOTAL_S });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, reset, say]);

  return (
    <div id="flex-guide" className="panel absolute bottom-[392px] left-4 w-64 space-y-2 p-3">
      <div className="flex items-center justify-between">
        <span className="t-section">Flex guide</span>
        <span className="chip">
          {info.rep > 0 ? `Rep ${info.rep}/${REPS} · ${info.phase}` : info.phase}
        </span>
      </div>
      <div ref={mountRef} className="h-[150px] w-full overflow-hidden rounded-md" />
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-(--line)"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(info.progress * 100)}
      >
        <div
          className="h-full bg-(--primary)"
          style={{ width: `${info.progress * 100}%` }}
        />
      </div>
      <div className="flex gap-2">
        {playing ? (
          <button className="btn btn-danger flex-1" onClick={reset}>
            Stop
          </button>
        ) : (
          <button className="btn btn-primary flex-1" onClick={() => setPlaying(true)}>
            Play
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
    </div>
  );
}
