/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import loadMujoco from "@mujoco/mujoco";

const MODEL_DIR = "/models/myohand";
const MODEL_FILE = "myohand_r.xml";
const FLEXOR = /^(FDS|FDP|FPL|FCR|FCU|PL$|LU_RB|UI_UB|RI)/;
const EXTENSOR = /^(EDC|EDM|EIP|EPL|EPB|ECR|ECU|APL)/;
// Highlighted muscle groups: [name pattern, resting colour, fully-active colour].
// Group 0 = superficial flexor (FDS), group 1 = deep flexor (FDP); every other
// muscle falls into the grey, semi-transparent layer (group 2).
const HIGHLIGHT = [
  { label: "Superficial flexor (FDS)", re: /^FDS/, rest: 0xd9892b, active: 0xffd23f },
  { label: "Deep flexor (FDP)", re: /^FDP/, rest: 0x1f9e9a, active: 0x4df5e8 },
];
const GREY = 0x9aa0a6;
const MJ_TYPE_MESH = 7;
// Joint-tracking gains (tuned headless against the MyoHand index finger).
// Forearm pose: -1.5 rad on pro_sup_r = supination (palm up; 0 is neutral, +ve is pronation).
const SUPINATION = -1.5;
const KP = 0.2;
const KP_PIP = 0.2; // PIP follows 0.65 x the MCP target; without this PIP creeps into flexion in supination
const FLEX_MAX = 0.06;
const EXT_MAX = 0.25;
const MJ_WRAP_PULLEY = 2;

type Controls = {
  flexor: number;
  extensor: number;
  paused: boolean;
  reset: boolean;
};

export type MuscleDrive = Record<string, number>;
export type ViewerHooks = {
  /** When non-null, overrides all actuator controls by muscle name (others = 0). */
  drive?: RefObject<MuscleDrive | null>;
  /**
   * Closed-loop joint tracking: a P controller on `joint` sets flexor/extensor drive so the
   * joint follows `target` (rad). Open-loop drive is bistable in this model (snaps to a fist
   * and does not return), so tracking is what gives a smooth curl and release.
   */
  curl?: RefObject<{ joint: string; pip: string; flex: string[]; ext: string[]; target: number } | null>;
  /** Per-muscle healing 0..1 (red = damaged, green = healed); tints the tendon. */
  health?: RefObject<Record<string, number>>;
  /** Muscles to emphasise (pulsing colour); everything else is dimmed while non-empty. */
  highlight?: RefObject<{ muscles: Set<string>; color: number } | null>;
  /**
   * Screen anchors: the viewer writes the projected path midpoint (px, relative to the viewer)
   * and live activation of every muscle in `wanted` into `pos` each frame.
   */
  anchors?: RefObject<{ wanted: Set<string>; pos: Record<string, MuscleAnchor> }>;
};
export type MuscleAnchor = { x: number; y: number; visible: boolean; act: number };

export default function MyoViewer({ drive, curl, health, highlight, anchors }: ViewerHooks = {}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const ctl = useRef<Controls>({
    flexor: 0,
    extensor: 0,
    paused: false,
    reset: false,
  });
  const [status, setStatus] = useState("Loading MuJoCo…");
  // ?grasp starts with the Grasp preset applied (handy for screenshots).
  // The viewer is client-only (ssr: false), so reading location here is safe.
  const [startGrasp] = useState(() =>
    new URLSearchParams(location.search).has("grasp"),
  );
  const [flexor, setFlexor] = useState(startGrasp ? 0.6 : 0);
  const [extensor, setExtensor] = useState(startGrasp ? 0.05 : 0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const mount = mountRef.current!;
    if (startGrasp) {
      ctl.current.flexor = 0.6;
      ctl.current.extensor = 0.05;
    }
    let cancelled = false;
    let cleanup = () => {};

    (async () => {
      const mujoco: any = await loadMujoco();
      if (cancelled) return;

      // Mirror the model files into MuJoCo's virtual filesystem.
      setStatus("Fetching model…");
      const xml = await (await fetch(`${MODEL_DIR}/${MODEL_FILE}`)).text();
      const meshFiles = Array.from(
        new DOMParser()
          .parseFromString(xml, "text/xml")
          .querySelectorAll("mesh[file]"),
      ).map((m) => m.getAttribute("file")!);
      mujoco.FS.mkdirTree("/model/meshes");
      mujoco.FS.writeFile(`/model/${MODEL_FILE}`, xml);
      await Promise.all(
        meshFiles.map(async (f) => {
          const buf = await (await fetch(`${MODEL_DIR}/meshes/${f}`)).arrayBuffer();
          mujoco.FS.writeFile(`/model/meshes/${f}`, new Uint8Array(buf));
        }),
      );
      if (cancelled) return;

      const model: any = mujoco.MjModel.from_xml_path(`/model/${MODEL_FILE}`);
      const data: any = new mujoco.MjData(model);
      // Pin the forearm in supination: nothing in the model holds pro_sup_r against gravity.
      // The wrist is pinned neutral too (flexion 0, deviation 0) so the back of the hand is
      // straight and in line with the forearm instead of sagging.
      const pins = ([["pro_sup_r", SUPINATION], ["flexion_r", 0], ["deviation_r", 0]] as const).map(([n, v]) => {
        const j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n);
        return { q: model.jnt_qposadr[j] as number, v: model.jnt_dofadr[j] as number, val: v as number };
      });
      const pinPose = () => {
        for (const p of pins) {
          data.qpos[p.q] = p.val;
          data.qvel[p.v] = 0;
        }
      };
      pinPose();
      mujoco.mj_forward(model, data);

      // --- three.js scene (MuJoCo is Z-up) ---
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0d12);
      const camera = new THREE.PerspectiveCamera(
        40,
        mount.clientWidth / mount.clientHeight,
        0.01,
        20,
      );
      camera.up.set(0, 0, 1);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      mount.appendChild(renderer.domElement);
      scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.6));
      const sun = new THREE.DirectionalLight(0xffffff, 2);
      sun.position.set(1, -1, 2);
      scene.add(sun);

      // Only show the forearm + hand (descendants of the radius/ulna).
      const roots = [model.body("radius_r").id, model.body("ulna_r").id];
      const inArm = (b: number) => {
        while (b > 0) {
          if (roots.includes(b)) return true;
          b = model.body_parentid[b];
        }
        return false;
      };

      const boneMat = new THREE.MeshStandardMaterial({
        color: 0xe8dcc4,
        roughness: 0.6,
      });
      const geomMeshes: { id: number; obj: THREE.Mesh }[] = [];
      const geometries: THREE.BufferGeometry[] = [];
      for (let g = 0; g < model.ngeom; g++) {
        if (model.geom_type[g] !== MJ_TYPE_MESH) continue;
        if (!inArm(model.geom_bodyid[g])) continue;
        const m = model.geom_dataid[g];
        const va = model.mesh_vertadr[m];
        const vn = model.mesh_vertnum[m];
        const fa = model.mesh_faceadr[m];
        const fn = model.mesh_facenum[m];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute(
          "position",
          new THREE.BufferAttribute(
            new Float32Array(model.mesh_vert.subarray(3 * va, 3 * (va + vn))),
            3,
          ),
        );
        geo.setIndex(
          new THREE.BufferAttribute(
            new Uint32Array(model.mesh_face.subarray(3 * fa, 3 * (fa + fn))),
            1,
          ),
        );
        geo.computeVertexNormals();
        const obj = new THREE.Mesh(geo, boneMat);
        obj.matrixAutoUpdate = false;
        scene.add(obj);
        geometries.push(geo);
        geomMeshes.push({ id: g, obj });
      }

      // Muscles: one tendon polyline each, colored by activation.
      // Drawn as shaded tubes: one instanced capsule per path segment plus a
      // sphere at each bend so the joints stay smooth while the path changes.
      const tendons: { id: number; act: number }[] = [];
      const names: string[] = [];
      for (let a = 0; a < model.nu; a++) {
        const tid = model.actuator_trnid[2 * a];
        names.push(
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, a),
        );
        tendons.push({ id: tid, act: a });
      }
      const MAX_INST = 8192;
      // Peak force (gainprm[2], stride 10) stands in for physiological
      // cross-section: stronger muscles get a thicker belly.
      const force: number[] = tendons.map(({ act }) =>
        Math.abs(model.actuator_gainprm[10 * act + 2]) || 1,
      );
      const maxForce = Math.max(...force);
      const pts: THREE.Vector3[] = [];
      const brk: boolean[] = [];
      const tubeMat = new THREE.MeshStandardMaterial({
        roughness: 0.4,
        metalness: 0,
      });
      // Other muscles: grey and see-through so the highlighted flexors read.
      const greyMat = new THREE.MeshStandardMaterial({
        roughness: 0.6,
        metalness: 0,
        transparent: true,
        opacity: 0.28,
        depthWrite: false,
      });
      const segGeo = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);
      const jointGeo = new THREE.SphereGeometry(1, 10, 8);
      // Index 0 = highlighted (opaque), 1 = grey (transparent).
      const layers = [tubeMat, greyMat].map((mat) => ({
        segs: new THREE.InstancedMesh(segGeo, mat, MAX_INST),
        joints: new THREE.InstancedMesh(jointGeo, mat, MAX_INST),
        ns: 0,
        nj: 0,
      }));
      for (const L of layers) {
        for (const m of [L.segs, L.joints]) {
          m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
          m.frustumCulled = false;
          m.count = 0;
          scene.add(m);
        }
      }
      const groupOf = names.map((n) => HIGHLIGHT.findIndex((g) => g.re.test(n)));
      const restCols = HIGHLIGHT.map((g) => new THREE.Color(g.rest));
      const activeCols = HIGHLIGHT.map((g) => new THREE.Color(g.active));
      const greyCol = new THREE.Color(GREY);
      const white = new THREE.Color(0xffffff);
      const anchorV = new THREE.Vector3();
      const hsl = { h: 0, s: 0, l: 0 };
      const pA = new THREE.Vector3();
      const dir = new THREE.Vector3();
      const mid = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      const scl = new THREE.Vector3();
      const yAxis = new THREE.Vector3(0, 1, 0);
      const iMat = new THREE.Matrix4();
      const flexIdx = names.flatMap((n, i) => (FLEXOR.test(n) ? [i] : []));
      const extIdx = names.flatMap((n, i) => (EXTENSOR.test(n) ? [i] : []));

      const target = new THREE.Vector3();
      const tb = model.body("capitate_r").id;
      target.set(data.xpos[3 * tb], data.xpos[3 * tb + 1], data.xpos[3 * tb + 2]);
      camera.position.copy(target).add(new THREE.Vector3(0.0, -0.3, 0.38)); // from above: palm faces up
      const orbit = new OrbitControls(camera, renderer.domElement);
      orbit.target.copy(target);
      orbit.update();

      const onResize = () => {
        camera.aspect = mount.clientWidth / mount.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mount.clientWidth, mount.clientHeight);
      };
      const ro = new ResizeObserver(onResize);
      ro.observe(mount);

      const tmp = new THREE.Color();
      const mat4 = new THREE.Matrix4();
      const dt = model.opt.timestep as number;
      let last = performance.now();
      let acc = 0;
      let raf = 0;
      setStatus("");

      const frame = (now: number) => {
        raf = requestAnimationFrame(frame);
        const c = ctl.current;
        if (c.reset) {
          mujoco.mj_resetData(model, data);
          pinPose();
          c.reset = false;
        }
        const d = drive?.current;
        const cv = curl?.current;
        if (cv) {
          const jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, cv.joint);
          const q = data.qpos[model.jnt_qposadr[jid]];
          const qp = data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, cv.pip)]];
          const u = KP * (cv.target - q) + KP_PIP * (0.65 * cv.target - qp);
          const f = Math.min(Math.max(u, 0), FLEX_MAX);
          const e = Math.min(Math.max(-4 * u, 0), EXT_MAX);
          data.ctrl.fill(0);
          for (const n of cv.flex) data.ctrl[names.indexOf(n)] = f;
          for (const n of cv.ext) data.ctrl[names.indexOf(n)] = e;
        } else if (d) {
          data.ctrl.fill(0);
          for (const [n, v] of Object.entries(d)) {
            const i = names.indexOf(n);
            if (i >= 0) data.ctrl[i] = v;
          }
        } else {
          for (const i of flexIdx) data.ctrl[i] = c.flexor;
          for (const i of extIdx) data.ctrl[i] = c.extensor;
        }
        acc += Math.min((now - last) / 1000, 0.05);
        last = now;
        if (!c.paused) {
          let n = 0;
          while (acc >= dt && n++ < 40) {
            pinPose();
            mujoco.mj_step(model, data);
            acc -= dt;
          }
        }
        acc = Math.min(acc, dt);
        mujoco.mj_forward(model, data);

        for (const { id, obj } of geomMeshes) {
          const p = data.geom_xpos;
          const R = data.geom_xmat;
          mat4.set(
            R[9 * id], R[9 * id + 1], R[9 * id + 2], p[3 * id],
            R[9 * id + 3], R[9 * id + 4], R[9 * id + 5], p[3 * id + 1],
            R[9 * id + 6], R[9 * id + 7], R[9 * id + 8], p[3 * id + 2],
            0, 0, 0, 1,
          );
          obj.matrix.copy(mat4);
        }
        for (const L of layers) {
          L.ns = 0;
          L.nj = 0;
        }
        for (const { id, act } of tendons) {
          const grp = groupOf[act];
          const hl = highlight?.current;
          const hlOn = !!hl && hl.muscles.size > 0;
          const isHl = hlOn && hl!.muscles.has(names[act]);
          // While something is highlighted, every other muscle moves to the translucent layer.
          const L = layers[isHl || (!hlOn && grp >= 0) ? 0 : 1];
          const adr = data.ten_wrapadr[id];
          const n = data.ten_wrapnum[id];
          const aa = model.actuator_actadr[act];
          const level = Math.min(
            Math.max(aa >= 0 ? data.act[aa] : data.ctrl[act], 0),
            1,
          );
          const h = health?.current?.[names[act]];
          if (grp >= 0) {
            tmp.copy(restCols[grp]).lerp(activeCols[grp], level);
            if (h !== undefined) {
              // Keep the hue (that is the label); show damage as dull and dark.
              const hh = Math.min(Math.max(h, 0), 1);
              tmp.getHSL(hsl);
              tmp.setHSL(hsl.h, hsl.s * (0.2 + 0.8 * hh), hsl.l * (0.5 + 0.5 * hh));
            }
          } else tmp.copy(greyCol);
          if (isHl) {
            const pulse = 0.5 + 0.5 * Math.sin(now / 180);
            tmp.setHex(hl!.color).lerp(white, 0.15 + 0.45 * pulse);
          }
          // Spindle profile: thin tendon at both ends, a belly in the middle
          // sized by peak force, swelling further with activation.
          const rEnd = 0.0005;
          const rMid =
            (0.0012 + 0.0026 * Math.sqrt(force[act] / maxForce)) *
            (1 + 0.35 * level);
          // Gather the path, then total its connected length.
          const m = Math.min(n, 256);
          for (let k = 0; k < m; k++) {
            const w = 3 * (adr + k);
            (pts[k] ??= new THREE.Vector3()).set(
              data.wrap_xpos[w], data.wrap_xpos[w + 1], data.wrap_xpos[w + 2],
            );
            brk[k] = k === 0 || data.wrap_obj[adr + k] === -2;
          }
          let total = 0;
          for (let k = 1; k < m; k++) if (!brk[k]) total += pts[k].distanceTo(pts[k - 1]);
          if (total < 1e-6) continue;
          const an = anchors?.current;
          if (an && an.wanted.has(names[act])) {
            // Point at half the connected path length.
            let walked = 0;
            let found = false;
            for (let k = 1; k < m && !found; k++) {
              if (brk[k]) continue;
              const len = pts[k].distanceTo(pts[k - 1]);
              if (walked + len >= total / 2) {
                anchorV.lerpVectors(pts[k - 1], pts[k], len > 0 ? (total / 2 - walked) / len : 0);
                found = true;
              }
              walked += len;
            }
            if (found) {
              anchorV.project(camera);
              an.pos[names[act]] = {
                x: ((anchorV.x + 1) / 2) * mount.clientWidth,
                y: ((1 - anchorV.y) / 2) * mount.clientHeight,
                visible: anchorV.z < 1 && Math.abs(anchorV.x) <= 1.2 && Math.abs(anchorV.y) <= 1.2,
                act: level,
              };
            }
          }
          const radiusAt = (d: number) => {
            const s = Math.sin(Math.PI * Math.min(Math.max(d / total, 0), 1));
            return rEnd + (rMid - rEnd) * Math.pow(s, 0.8);
          };
          // Split each segment into ~3 mm pieces so the profile is smooth.
          let dist = 0;
          for (let k = 1; k < m && L.ns < MAX_INST - 64; k++) {
            if (brk[k]) continue;
            const len = pts[k].distanceTo(pts[k - 1]);
            if (len < 1e-6) continue;
            dir.subVectors(pts[k], pts[k - 1]).divideScalar(len);
            quat.setFromUnitVectors(yAxis, dir);
            const pieces = Math.max(1, Math.ceil(len / 0.003));
            const step = len / pieces;
            for (let i = 0; i < pieces; i++) {
              const d0 = dist + i * step;
              const rr = radiusAt(d0 + step / 2);
              pA.copy(pts[k - 1]).addScaledVector(dir, i * step);
              mid.copy(pA).addScaledVector(dir, step / 2);
              iMat.compose(mid, quat, scl.set(rr, step * 1.02, rr));
              L.segs.setMatrixAt(L.ns, iMat);
              L.segs.setColorAt(L.ns, tmp);
              L.ns++;
            }
            // Round joint only at real path bends (and path ends).
            const rb = radiusAt(dist + len);
            iMat.compose(pts[k], quat.identity(), scl.set(rb, rb, rb));
            L.joints.setMatrixAt(L.nj, iMat);
            L.joints.setColorAt(L.nj, tmp);
            L.nj++;
            dist += len;
          }
        }
        for (const L of layers) {
          L.segs.count = L.ns;
          L.joints.count = L.nj;
          for (const m of [L.segs, L.joints]) {
            m.instanceMatrix.needsUpdate = true;
            if (m.instanceColor) m.instanceColor.needsUpdate = true;
          }
        }
        orbit.update();
        renderer.render(scene, camera);
      };
      raf = requestAnimationFrame(frame);

      cleanup = () => {
        cancelAnimationFrame(raf);
        ro.disconnect();
        orbit.dispose();
        geometries.forEach((g) => g.dispose());
        for (const L of layers) {
          L.segs.dispose();
          L.joints.dispose();
        }
        segGeo.dispose();
        jointGeo.dispose();
        tubeMat.dispose();
        greyMat.dispose();
        boneMat.dispose();
        renderer.dispose();
        renderer.domElement.remove();
        data.delete();
        model.delete();
      };
      void MJ_WRAP_PULLEY;
    })().catch((e) => {
      console.error(e);
      setStatus(`Failed to load: ${e?.message ?? e}`);
    });

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [startGrasp, drive, curl, health, highlight, anchors]);

  const set = (patch: Partial<Controls>) => Object.assign(ctl.current, patch);

  return (
    <div className="relative h-full w-full">
      <div ref={mountRef} className="h-full w-full" />
      {status && (
        <div className="absolute inset-0 flex items-center justify-center muted">
          {status}
        </div>
      )}
      <div className="panel absolute left-4 top-4 w-64 space-y-3">
        <div className="t-title">MyoHand (MuJoCo WASM)</div>
        <ul className="space-y-1.5">
          {[
            ...HIGHLIGHT.map((g) => ({ label: g.label, color: g.rest })),
            { label: "Other muscles", color: GREY },
          ].map((k) => (
            <li key={k.label} className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-5 rounded-full"
                style={{
                  background: `#${k.color.toString(16).padStart(6, "0")}`,
                  opacity: k.color === GREY ? 0.5 : 1,
                }}
              />
              {k.label}
            </li>
          ))}
        </ul>
        <label className="t-body block">
          Flexors: {flexor.toFixed(2)}
          <input
            type="range" min={0} max={1} step={0.01} value={flexor}
            className="w-full"
            onChange={(e) => { const v = +e.target.value; setFlexor(v); set({ flexor: v }); }}
          />
        </label>
        <label className="t-body block">
          Extensors: {extensor.toFixed(2)}
          <input
            type="range" min={0} max={1} step={0.01} value={extensor}
            className="w-full"
            onChange={(e) => { const v = +e.target.value; setExtensor(v); set({ extensor: v }); }}
          />
        </label>
        <div className="flex gap-2">
          <button
            className="btn"
            onClick={() => { setPaused(!paused); set({ paused: !paused }); }}
          >
            {paused ? "Play" : "Pause"}
          </button>
          <button
            className="btn"
            onClick={() => {
              setFlexor(0); setExtensor(0);
              set({ flexor: 0, extensor: 0, reset: true });
            }}
          >
            Reset
          </button>
          <button
            className="btn"
            onClick={() => {
              setFlexor(0.6); setExtensor(0.05);
              set({ flexor: 0.6, extensor: 0.05 });
            }}
          >
            Grasp
          </button>
        </div>
      </div>
    </div>
  );
}
