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
  showForce: boolean;
};

export type MuscleDrive = Record<string, number>;
/**
 * What the view cube can ask of the main camera. The camera orbits a fixed point, so every call
 * here is a rotation about it: nothing zooms or pans.
 */
export type ViewApi = {
  /** The camera's world orientation, which the cube mirrors (inverted) to show the world axes. */
  quaternion: () => THREE.Quaternion;
  /** Fly to look at the hand from along `dir` (a world axis, the camera sits on that side) with `up` up. */
  snapTo: (dir: readonly [number, number, number], up: readonly [number, number, number]) => void;
  /** Turn 90 degrees (or any angle) about a camera axis: yaw about up, pitch about right, roll about the view axis. */
  turn: (axis: "yaw" | "pitch" | "roll", radians: number) => void;
  /** Rotate freely while dragging: `dx` about the camera's up axis, `dy` about its right axis (radians). */
  drag: (dx: number, dy: number) => void;
  /** End of a drag: hand the camera back to the orbit controls. */
  commit: () => void;
  /** Back to the starting side profile. */
  home: () => void;
  /**
   * Swing the camera round to the side of the arm that `muscle` is on, so it faces the tendon. The
   * orbit centre is fixed, so this is a rotation only. The muscle's position is read on the next
   * rendered frame; an unknown name does nothing. `ms` is the duration of the move.
   */
  focusMuscle: (muscle: string, ms?: number) => void;
};

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
  /** Fingertip force (N) during a live curl; drawn as an arrow at the index fingertip. Null hides it. */
  force?: RefObject<{ n: number } | null>;
  /**
   * Hand-tracking mode: when non-null, these joint angles (rad, by joint name) are imposed
   * kinematically every step and all muscle drive is off, so the sim hand mirrors a tracked hand.
   */
  handPose?: RefObject<Record<string, number> | null>;
  /** Explore mode: while true, wheel zoom is disabled so the wheel can step through tendons. */
  explore?: RefObject<boolean>;
  /** The Explore toggle shown in the top-left card. Omit `onExploreToggle` to hide it. */
  exploreOn?: boolean;
  /** Which tendon explore mode is on, for example "3/20 Deep flexor (FDP)". */
  exploreLabel?: string | null;
  onExploreToggle?: () => void;
  /** Called with the camera API for the view cube once the viewer is ready, and with null when it goes away. */
  onViewApi?: (api: ViewApi | null) => void;
};
// Force arrow: length grows linearly with force, from the index fingertip, pointing out along the finger in the xy-plane.
const FORCE_ARROW_MIN = 0.008; // m
const FORCE_ARROW_PER_N = 0.0022; // m per newton (28 N peak is about 7 cm)
const FORCE_HEAD_WIDTH = 0.007; // m
export type MuscleAnchor = { x: number; y: number; visible: boolean; act: number };

export default function MyoViewer({ drive, curl, health, highlight, anchors, force: forceRef, handPose, explore, exploreOn, exploreLabel, onExploreToggle, onViewApi }: ViewerHooks = {}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const ctl = useRef<Controls>({
    flexor: 0,
    extensor: 0,
    paused: false,
    reset: false,
    showForce: true,
  });
  const [showForce, setShowForce] = useState(true);
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
      // While a curl is played, every other digit is held in extension (flexion joints at 0).
      const restJoints = [
        ...[3, 4, 5].flatMap((d) => [`mcp${d}_flexion_r`, `pm${d}_flexion_r`, `md${d}_flexion_r`]),
        "cmc_flexion_r", "mp_flexion_r", "ip_flexion_r",
      ];
      const restPins = restJoints.map((n) => {
        const j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n);
        return { q: model.jnt_qposadr[j] as number, v: model.jnt_dofadr[j] as number };
      });
      const jointIds = new Map<string, number>();
      const jointId = (n: string) => {
        if (!jointIds.has(n)) jointIds.set(n, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n));
        return jointIds.get(n)!;
      };
      const pinPose = () => {
        const hp = handPose?.current;
        if (hp) {
          for (const [n, val] of Object.entries(hp)) {
            const j = jointId(n);
            if (j < 0) continue;
            const lo = model.jnt_range[2 * j];
            const hi = model.jnt_range[2 * j + 1];
            data.qpos[model.jnt_qposadr[j]] = Math.min(Math.max(val, lo), hi);
            data.qvel[model.jnt_dofadr[j]] = 0;
          }
        } else if (curl?.current) {
          for (const p of restPins) {
            data.qpos[p.q] = 0;
            data.qvel[p.v] = 0;
          }
        }
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

      const tipSite = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, "IFtip_r");
      const forceArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.05, 0xff6b6b, 0.012, FORCE_HEAD_WIDTH);
      const midBody = model.body("midph2_r").id;
      const forceDir = new THREE.Vector3();
      forceArrow.visible = false;
      forceArrow.renderOrder = 10;
      for (const m of [forceArrow.line.material, forceArrow.cone.material] as THREE.Material[]) {
        m.depthTest = false;
      }
      scene.add(forceArrow);

      const target = new THREE.Vector3();
      const tb =model.body("capitate_r").id;
      target.set(data.xpos[3 * tb], data.xpos[3 * tb + 1], data.xpos[3 * tb + 2]);
      // Side profile: the forearm runs along Y and the fingers curl in the Y-Z plane (palm faces +Z),
      // so looking along X, level with the hand, shows the flexion in profile. Z is the camera's up
      // axis, so the fingers point to the right of the screen. The orbit controls still let you turn it.
      // Raised 0.15 so the camera looks down on the hand at about 15 degrees. Both the camera and
      // the point it looks at are moved 0.04 along +Y (screen right), which slides the arm left
      // of the screen centre without turning the view away from side-on.
      const SHIFT_RIGHT = 0.04;
      const lookAt = target.clone().add(new THREE.Vector3(0.0, SHIFT_RIGHT, 0.0));
      const finalOffset = new THREE.Vector3(0.56, 0.0, 0.15);
      camera.position.copy(lookAt).add(finalOffset);
      // Replaceable: OrbitControls fixes its notion of "up" when it is built, so rolling the view (see
      // the view cube below) means building a fresh one that picks up the camera's new up axis.
      let orbit = new OrbitControls(camera, renderer.domElement);
      orbit.target.copy(lookAt);
      orbit.update();

      // Intro: for the first 10 s the camera drifts round the hand and slows into the pose above.
      // It starts where the old view was (behind and above the hand, looking along the forearm,
      // centred on the hand) and sweeps 90 degrees round to the side profile while the centre
      // slides to the left-offset point, decelerating so it settles rather than stops. Touching
      // the controls ends it at once so it never fights the user, and a reduced-motion preference
      // skips it.
      const INTRO_MS = 10000;
      const finalR = finalOffset.length();
      const finalAz = Math.atan2(finalOffset.y, finalOffset.x);
      const finalEl = Math.asin(finalOffset.z / finalR);
      const introFrom = { az: finalAz - Math.PI / 2, el: 0.9, r: 0.5 };
      const centre = new THREE.Vector3();
      const placeCamera = (e: number) => {
        const az = introFrom.az + (finalAz - introFrom.az) * e;
        const el = introFrom.el + (finalEl - introFrom.el) * e;
        const r = introFrom.r + (finalR - introFrom.r) * e;
        centre.lerpVectors(target, lookAt, e);
        orbit.target.copy(centre);
        camera.position.set(
          centre.x + r * Math.cos(el) * Math.cos(az),
          centre.y + r * Math.cos(el) * Math.sin(az),
          centre.z + r * Math.sin(el),
        );
      };
      let introActive = !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      let introStart: number | null = null;
      if (introActive) placeCamera(0); // so the first frame is the start of the sweep, not a jump
      const stopIntro = () => { introActive = false; };
      orbit.addEventListener("start", stopIntro);

      // --- View cube support. While `viewBusy` the cube is moving the camera by hand and the orbit
      // controls are left alone, so the two never fight. Orientation is set directly as a
      // quaternion (the camera looks along its own -Z), so there is no gimbal lock and no limit
      // at the poles, which is what lets the cube rotate the view about every axis.
      let viewBusy = false;
      let viewAnim: { qs: THREE.Quaternion; qe: THREE.Quaternion; r: number; t0: number; ms: number } | null = null;
      const zero = new THREE.Vector3();
      const syncOrbit = () => {
        const t = orbit.target.clone();
        orbit.dispose();
        orbit = new OrbitControls(camera, renderer.domElement);
        orbit.target.copy(t);
        orbit.addEventListener("start", stopIntro);
        orbit.update();
      };
      const orient = (dir: THREE.Vector3, up: THREE.Vector3) =>
        new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(dir, zero, up));
      const applyOrientation = (q: THREE.Quaternion, r: number) => {
        camera.quaternion.copy(q);
        camera.up.set(0, 1, 0).applyQuaternion(q);
        camera.position.copy(orbit.target).add(new THREE.Vector3(0, 0, r).applyQuaternion(q));
      };
      const animateTo = (qe: THREE.Quaternion, r?: number, ms = 450) => {
        stopIntro();
        viewBusy = true;
        viewAnim = {
          qs: camera.quaternion.clone(), qe, t0: performance.now(), ms,
          r: r ?? camera.position.distanceTo(orbit.target),
        };
      };
      const axes = { yaw: new THREE.Vector3(0, 1, 0), pitch: new THREE.Vector3(1, 0, 0), roll: new THREE.Vector3(0, 0, 1) };
      // A pending "face this muscle" request from the chat agent, resolved in the render loop once the
      // muscle's path midpoint is known (see where the anchors are computed).
      let focusReq: { name: string; ms: number } | null = null;
      const reducedMotion = !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      const focusOnPoint = (p: THREE.Vector3, ms: number) => {
        // The forearm runs along Y, so a point's offset along Y says where it is on the arm and says
        // nothing about which side of it to view from. Look from the point's side of the arm (the
        // part of the offset across Y), leaning a little along the arm so the tendon is not edge-on.
        const off = p.clone().sub(orbit.target);
        const side = new THREE.Vector3(off.x, 0, off.z);
        if (side.lengthSq() < 1e-8) side.copy(camera.position).sub(orbit.target).setY(0);
        if (side.lengthSq() < 1e-8) side.set(1, 0, 0);
        side.normalize().multiplyScalar(0.9).add(new THREE.Vector3(0, Math.sign(off.y || 1) * 0.25, 0.15)).normalize();
        // Z is up, except looking straight along it (the palm or back of the hand), where X is.
        const up = Math.abs(side.z) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
        animateTo(orient(side, up), undefined, reducedMotion ? 1 : ms);
      };
      const stepViewAnimation = (now: number) => {
        if (!viewAnim) return;
        const u = Math.min((now - viewAnim.t0) / viewAnim.ms, 1);
        const e = u * u * (3 - 2 * u);
        applyOrientation(viewAnim.qs.clone().slerp(viewAnim.qe, e), viewAnim.r);
        if (u >= 1) {
          viewAnim = null;
          viewBusy = false;
          syncOrbit();
        }
      };
      {
        onViewApi?.({
          quaternion: () => camera.quaternion,
          snapTo: (dir, up) => animateTo(orient(new THREE.Vector3(...dir), new THREE.Vector3(...up))),
          turn: (axis, radians) => {
            // Build on where a running animation is heading, so quick repeated presses add up.
            const from = viewAnim ? viewAnim.qe : camera.quaternion;
            animateTo(from.clone().multiply(new THREE.Quaternion().setFromAxisAngle(axes[axis], radians)));
          },
          drag: (dx, dy) => {
            stopIntro();
            viewAnim = null;
            viewBusy = true;
            const q = camera.quaternion.clone()
              .multiply(new THREE.Quaternion().setFromAxisAngle(axes.yaw, -dx))
              .multiply(new THREE.Quaternion().setFromAxisAngle(axes.pitch, -dy));
            applyOrientation(q, camera.position.distanceTo(orbit.target));
          },
          commit: () => {
            if (!viewBusy || viewAnim) return;
            viewBusy = false;
            syncOrbit();
          },
          home: () => animateTo(orient(finalOffset, new THREE.Vector3(0, 0, 1)), finalR),
          focusMuscle: (muscle, ms = 900) => { focusReq = { name: muscle, ms }; },
        });
      }

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
        if (handPose?.current) {
          data.ctrl.fill(0);
        } else if (cv) {
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
          const wantAnchor = !!an && an.wanted.has(names[act]);
          const wantFocus = focusReq?.name === names[act];
          if (wantAnchor || wantFocus) {
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
            if (found && wantFocus && focusReq) {
              focusOnPoint(anchorV, focusReq.ms);
              focusReq = null;
            }
            if (found && wantAnchor && an) {
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
        const fv = forceRef?.current;
        if (fv && tipSite >= 0 && c.showForce) {
          const len = FORCE_ARROW_MIN + FORCE_ARROW_PER_N * Math.max(fv.n, 0);
          forceArrow.position.set(
            data.site_xpos[3 * tipSite], data.site_xpos[3 * tipSite + 1], data.site_xpos[3 * tipSite + 2],
          );
          // Outward along the finger (middle phalanx -> tip), flattened to the xy-plane: the
          // resisting pull opposes the curl.
          forceDir.set(
            data.site_xpos[3 * tipSite] - data.xpos[3 * midBody],
            data.site_xpos[3 * tipSite + 1] - data.xpos[3 * midBody + 1],
            0,
          );
          if (forceDir.lengthSq() > 1e-10) forceArrow.setDirection(forceDir.normalize());
          forceArrow.setLength(len, Math.min(0.012, len * 0.4), FORCE_HEAD_WIDTH);
          forceArrow.visible = true;
        } else forceArrow.visible = false;
        if (introActive) {
          introStart ??= now;
          const u = Math.min((now - introStart) / INTRO_MS, 1);
          placeCamera(1 - (1 - u) * (1 - u)); // ease out: fastest at the start, settling at the end
          if (u >= 1) introActive = false;
        }
        stepViewAnimation(now);
        if (!viewBusy) {
          orbit.enableZoom = !explore?.current;
          orbit.update();
        }
        renderer.render(scene, camera);
      };
      raf = requestAnimationFrame(frame);

      cleanup = () => {
        cancelAnimationFrame(raf);
        ro.disconnect();
        onViewApi?.(null);
        orbit.dispose();
        geometries.forEach((g) => g.dispose());
        for (const L of layers) {
          L.segs.dispose();
          L.joints.dispose();
        }
        scene.remove(forceArrow);
        forceArrow.dispose();
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
  }, [startGrasp, drive, curl, health, highlight, anchors, forceRef, handPose, explore, onViewApi]);

  const set = (patch: Partial<Controls>) => Object.assign(ctl.current, patch);

  // Collapse the sliders/buttons (keep title + legend) when they would overlap the flex guide panel.
  const panelRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [compact, setCompact] = useState(false);
  // The flexor/extensor drive sliders are hidden until asked for; the tracked hand and the sim run without them.
  const [showSliders, setShowSliders] = useState(false);
  useEffect(() => {
    const check = () => {
      const panel = panelRef.current;
      const body = bodyRef.current;
      const guide = document.getElementById("flex-guide");
      if (!panel || !body || !guide) { setCompact(false); return; }
      // Bottom edge the panel would have if fully expanded.
      const fullBottom = panel.getBoundingClientRect().bottom - body.offsetHeight + body.scrollHeight;
      setCompact(fullBottom + 8 > guide.getBoundingClientRect().top);
    };
    check();
    const id = window.setInterval(check, 400);
    window.addEventListener("resize", check);
    return () => { window.clearInterval(id); window.removeEventListener("resize", check); };
  }, []);

  return (
    <div className="relative h-full w-full">
      <div ref={mountRef} className="h-full w-full" />
      {status && (
        <div className="absolute inset-0 flex items-center justify-center muted">
          {status}
        </div>
      )}
      <div ref={panelRef} className="panel absolute left-4 top-4 w-[340px] max-w-[calc(100%-2rem)] space-y-3">
        <div className="t-title">Flex &apos;N Sense</div>
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
        <div
          ref={bodyRef}
          className="space-y-3 overflow-hidden transition-[max-height,opacity] duration-200"
          style={compact ? { maxHeight: 0, opacity: 0, marginTop: 0, pointerEvents: "none" } : { maxHeight: 400 }}
          aria-hidden={compact}
          inert={compact}
        >
        {showSliders && (
          <>
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
          </>
        )}
        <label className="t-body flex items-center gap-2">
          <input
            type="checkbox" checked={showForce}
            onChange={(e) => { setShowForce(e.target.checked); set({ showForce: e.target.checked }); }}
          />
          Show force vector
        </label>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            className="btn"
            hidden={compact}
            onClick={() => { setPaused(!paused); set({ paused: !paused }); }}
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            className="btn"
            onClick={() => {
              setFlexor(0); setExtensor(0);
              set({ flexor: 0, extensor: 0, reset: true });
            }}
          >
            Reset simulation
          </button>
          {onExploreToggle && (
            <button
              className={exploreOn ? "btn btn-primary" : "btn"}
              aria-pressed={!!exploreOn}
              title="Scroll the mouse wheel over the 3D view to step through every tendon (Esc to exit)"
              onClick={onExploreToggle}
            >
              Explore: {exploreOn ? "on" : "off"}
            </button>
          )}
          <button
            className="btn"
            hidden={compact}
            aria-pressed={showSliders}
            onClick={() => setShowSliders(!showSliders)}
          >
            Sliders: {showSliders ? "on" : "off"}
          </button>
          <button
            className="btn"
            hidden={compact}
            onClick={() => {
              setFlexor(0.6); setExtensor(0.05);
              set({ flexor: 0.6, extensor: 0.05 });
            }}
          >
            Grasp
          </button>
        </div>
        {exploreOn && (
          <p className="t-small muted">
            {/* The mouse wobbles for the first second of every 10 s cycle and rests for the other nine.
                The keyframes are inline so they travel with the component. */}
            <style>{`
              @keyframes explore-mouse-wobble {
                0% { transform: rotate(0deg); }
                2.5% { transform: rotate(-16deg); }
                5% { transform: rotate(14deg); }
                7.5% { transform: rotate(-9deg); }
                10%, 100% { transform: rotate(0deg); }
              }
              .explore-mouse { display: inline-block; transform-origin: 50% 80%; animation: explore-mouse-wobble 10s ease-in-out infinite; }
              @media (prefers-reduced-motion: reduce) { .explore-mouse { animation: none; } }
            `}</style>
            <span className="explore-mouse" aria-hidden="true">🖱️</span> Scroll to explore components
          </p>
        )}
        {exploreOn && exploreLabel && <p className="t-small muted">{exploreLabel}</p>}
        <p className="t-small faint">MyoHand Simulator</p>
      </div>
    </div>
  );
}
