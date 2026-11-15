"use client";

import { useEffect, useRef, type RefObject } from "react";
import * as THREE from "three";
import type { ViewApi } from "./MyoViewer";
import { CORNERS, FACES } from "../lib/viewPresets";

/**
 * A view cube in the style of FreeCAD's: it shows the world axes as the main camera sees them.
 *
 * - Click a face to fly the main camera to that view (Top, Bottom, Front, Back, Left, Right), or a
 *   corner to fly to the three-quarter view from that corner.
 * - Drag on the cube to rotate the view freely about every axis (no limit at the poles).
 * - The arrows step the view 90 degrees about its up and right axes, the curved arrows roll it
 *   90 degrees about the view axis, and the house returns to the starting side profile.
 *
 * It owns a tiny renderer of its own and reads the main camera's orientation every frame through
 * the `ViewApi` the viewer publishes, so the two cannot disagree.
 */

// How far a bevel cuts in along each of the three edges meeting at a corner, in cube units. The cube is
// about 55 px across on screen, so 0.2 makes each bevel a triangle roughly 11 px along its sides.
const BEVEL = 0.2;

const CANVAS_PX = 123;
const FRAME_PX = 167;
const BUTTON_PX = 26;
const STEP = Math.PI / 2;
/** Radians of rotation per pixel dragged on the cube. */
const DRAG_RAD_PER_PX = 0.012;
const DRAG_THRESHOLD_PX = 4;

function faceTexture(lines: string[], hover: boolean): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = hover ? "#4a5a70" : "#262d38";
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = hover ? "#9db4d2" : "#5d6978";
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, size - 8, size - 8);
    ctx.fillStyle = "#eceef1";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // The biggest font that keeps the widest line inside the border.
    let px = 26;
    do {
      ctx.font = `600 ${px}px system-ui, sans-serif`;
      px -= 1;
    } while (px > 8 && Math.max(...lines.map((l) => ctx.measureText(l).width)) > size - 28);
    const lineHeight = px * 1.35;
    lines.forEach((line, i) => {
      ctx.fillText(line, size / 2, size / 2 + (i - (lines.length - 1) / 2) * lineHeight + 1);
    });
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const BUTTONS: {
  key: string;
  glyph: string;
  title: string;
  pos: string;
  turn?: ["yaw" | "pitch" | "roll", number];
  home?: boolean;
}[] = [
  // The arrows move the scene in the direction they point, as FreeCAD's do.
  { key: "up", glyph: "▲", title: "Rotate the view up 90°", pos: "left-[70px] top-0", turn: ["pitch", STEP] },
  { key: "down", glyph: "▼", title: "Rotate the view down 90°", pos: "left-[70px] bottom-0", turn: ["pitch", -STEP] },
  { key: "left", glyph: "◀", title: "Rotate the view left 90°", pos: "left-0 top-[70px]", turn: ["yaw", STEP] },
  { key: "right", glyph: "▶", title: "Rotate the view right 90°", pos: "right-0 top-[70px]", turn: ["yaw", -STEP] },
  { key: "ccw", glyph: "↺", title: "Roll the view counterclockwise 90°", pos: "left-0 top-0", turn: ["roll", -STEP] },
  { key: "cw", glyph: "↻", title: "Roll the view clockwise 90°", pos: "right-0 top-0", turn: ["roll", STEP] },
  { key: "home", glyph: "⌂", title: "Back to the starting view", pos: "right-0 bottom-0", home: true },
];

export default function ViewCube({ api }: { api: RefObject<ViewApi | null> }) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(CANVAS_PX, CANVAS_PX);
    const dom = renderer.domElement;
    dom.style.cursor = "grab";
    dom.style.touchAction = "none";
    dom.style.display = "block";
    mount.appendChild(dom);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 20);
    camera.position.set(0, 0, 4.6);

    const cube = new THREE.Group();
    scene.add(cube);
    const zero = new THREE.Vector3();
    // Each face is a square with its four corners cut off by the bevel, so the bevels have room.
    const h = 0.5;
    const outline = [
      [-h + BEVEL, -h], [h - BEVEL, -h], [h, -h + BEVEL], [h, h - BEVEL],
      [h - BEVEL, h], [-h + BEVEL, h], [-h, h - BEVEL], [-h, -h + BEVEL],
    ];
    const faceShape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
    const faceGeometry = new THREE.ShapeGeometry(faceShape);
    {
      // ShapeGeometry's UVs are the raw coordinates (-0.5..0.5); the label texture wants 0..1.
      const pos = faceGeometry.getAttribute("position");
      const uv = faceGeometry.getAttribute("uv");
      for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) + 0.5, pos.getY(i) + 0.5);
      uv.needsUpdate = true;
    }
    const OUTLINE_COLOR = 0x8795a8;
    const outlineMaterial = new THREE.LineBasicMaterial({ color: OUTLINE_COLOR });
    const outlineGeometry = new THREE.BufferGeometry().setFromPoints(outline.map(([x, y]) => new THREE.Vector3(x, y, 0.002)));
    const planes = FACES.map((f, i) => {
      const normal = new THREE.MeshBasicMaterial({ map: faceTexture(f.lines, false) });
      const hover = faceTexture(f.lines, true);
      const mesh = new THREE.Mesh(faceGeometry, normal);
      // The cut corners are symmetric, so the outline can turn with the label without anyone noticing.
      mesh.add(new THREE.LineLoop(outlineGeometry, outlineMaterial));
      const dir = new THREE.Vector3(...f.dir);
      // The plane faces +Z with +Y up, so orienting it like a camera at `dir` puts its front on the
      // outside and its text upright for that view.
      mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().lookAt(dir, zero, new THREE.Vector3(...f.up)));
      mesh.position.copy(dir).multiplyScalar(0.5);
      mesh.userData.index = i;
      mesh.userData.hover = hover;
      mesh.userData.normal = normal.map;
      cube.add(mesh);
      return mesh;
    });
    // Corner bevels: a triangle across each vertex, joining the points BEVEL along the three edges that
    // meet there, so it fits exactly into the corner the faces leave open. They sit in the cube's group
    // so they turn with it, and only the ones on the near side can be hit (the others are hidden).
    const CORNER_COLOR = 0x4a5566;
    const CORNER_HOVER_COLOR = 0xf2c14e;
    const bevelGeometries: THREE.BufferGeometry[] = [];
    const corners = CORNERS.map((c, i) => {
      const [sx, sy, sz] = c.dir;
      const vx = sx * h, vy = sy * h, vz = sz * h;
      const a = new THREE.Vector3(vx - sx * BEVEL, vy, vz);
      const b = new THREE.Vector3(vx, vy - sy * BEVEL, vz);
      const d = new THREE.Vector3(vx, vy, vz - sz * BEVEL);
      // Wind it so the front side faces away from the cube's centre, where the viewer is.
      const outward = new THREE.Vector3(sx, sy, sz);
      const flip = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(d, a)).dot(outward) < 0;
      const points = flip ? [a, d, b] : [a, b, d];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      geometry.computeVertexNormals();
      bevelGeometries.push(geometry);
      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: CORNER_COLOR }));
      const rim = new THREE.BufferGeometry().setFromPoints(points);
      bevelGeometries.push(rim);
      mesh.add(new THREE.LineLoop(rim, outlineMaterial));
      mesh.userData.index = i;
      mesh.userData.corner = true;
      cube.add(mesh);
      return mesh;
    });

    // Keep every label legible: as the cube turns, each face spins its text in 90 degree steps so it
    // lies along the face edge that puts it closest to upright on screen. Steps of 90 degrees keep
    // the text aligned to an edge of the face, and a label only switches once another orientation
    // is clearly better, so it does not flicker where two are nearly tied.
    const baseQuat = planes.map((m) => m.quaternion.clone());
    const spin = [0, 1, 2, 3].map((k) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), (k * Math.PI) / 2));
    const spinOf = planes.map(() => 0);
    const textUp = new THREE.Vector3();
    const candidate = new THREE.Quaternion();
    const SWITCH_MARGIN = 0.2;
    const orientText = () => {
      planes.forEach((mesh, i) => {
        let best = spinOf[i];
        let bestScore = -Infinity;
        let currentScore = -Infinity;
        for (let k = 0; k < 4; k++) {
          candidate.copy(baseQuat[i]).multiply(spin[k]);
          // Where this orientation's "up" ends up on screen: the cube's own rotation is the only
          // thing between the face and the (fixed) display camera, whose up is +Y.
          textUp.set(0, 1, 0).applyQuaternion(candidate).applyQuaternion(cube.quaternion);
          if (k === spinOf[i]) currentScore = textUp.y;
          if (textUp.y > bestScore) {
            bestScore = textUp.y;
            best = k;
          }
        }
        if (best !== spinOf[i] && bestScore > currentScore + SWITCH_MARGIN) {
          spinOf[i] = best;
          mesh.quaternion.copy(baseQuat[i]).multiply(spin[best]);
        }
      });
    };

    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    type Target = { kind: "face" | "corner"; index: number };
    const pick = (e: PointerEvent): Target | null => {
      const r = dom.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      // Nearest thing under the pointer wins, so a corner on the near side beats the faces it sits
      // on, and a corner behind the cube loses to the face in front of it.
      const hit = raycaster.intersectObjects([...planes, ...corners], false)[0];
      if (!hit) return null;
      return { kind: hit.object.userData.corner ? "corner" : "face", index: hit.object.userData.index as number };
    };
    let hovered: Target | null = null;
    const setHover = (target: Target | null) => {
      if (hovered?.kind === target?.kind && hovered?.index === target?.index) return;
      if (hovered?.kind === "face") (planes[hovered.index].material as THREE.MeshBasicMaterial).map = planes[hovered.index].userData.normal;
      if (hovered?.kind === "corner") (corners[hovered.index].material as THREE.MeshBasicMaterial).color.setHex(CORNER_COLOR);
      hovered = target;
      if (target?.kind === "face") (planes[target.index].material as THREE.MeshBasicMaterial).map = planes[target.index].userData.hover;
      if (target?.kind === "corner") (corners[target.index].material as THREE.MeshBasicMaterial).color.setHex(CORNER_HOVER_COLOR);
      for (const p of planes) (p.material as THREE.MeshBasicMaterial).needsUpdate = true;
      dom.style.cursor = target === null ? "grab" : "pointer";
      dom.title = target?.kind === "corner" ? `View from the ${CORNERS[target.index].name} corner` : "";
    };

    // One gesture is either a click (snap to the face under the pointer) or a drag (rotate freely).
    let down: { x: number; y: number; lastX: number; lastY: number; dragging: boolean } | null = null;
    const onDown = (e: PointerEvent) => {
      dom.setPointerCapture(e.pointerId);
      down = { x: e.clientX, y: e.clientY, lastX: e.clientX, lastY: e.clientY, dragging: false };
    };
    const onMove = (e: PointerEvent) => {
      if (!down) {
        setHover(pick(e));
        return;
      }
      if (!down.dragging && Math.hypot(e.clientX - down.x, e.clientY - down.y) > DRAG_THRESHOLD_PX) {
        down.dragging = true;
        setHover(null);
        dom.style.cursor = "grabbing";
      }
      if (down.dragging) {
        api.current?.drag((e.clientX - down.lastX) * DRAG_RAD_PER_PX, (e.clientY - down.lastY) * DRAG_RAD_PER_PX);
        down.lastX = e.clientX;
        down.lastY = e.clientY;
      }
    };
    const onUp = (e: PointerEvent) => {
      const gesture = down;
      down = null;
      if (!gesture) return;
      if (gesture.dragging) {
        api.current?.commit();
        dom.style.cursor = "grab";
        return;
      }
      const target = pick(e);
      if (target?.kind === "face") api.current?.snapTo(FACES[target.index].dir, FACES[target.index].up);
      if (target?.kind === "corner") api.current?.snapTo(CORNERS[target.index].dir, CORNERS[target.index].up);
    };
    const onLeave = () => { if (!down) setHover(null); };
    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointermove", onMove);
    dom.addEventListener("pointerup", onUp);
    dom.addEventListener("pointercancel", onUp);
    dom.addEventListener("pointerleave", onLeave);

    let raf = 0;
    const tick = () => {
      const view = api.current;
      // The world as the camera sees it is the world turned by the camera's inverse rotation.
      if (view) cube.quaternion.copy(view.quaternion()).invert();
      orientText();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointermove", onMove);
      dom.removeEventListener("pointerup", onUp);
      dom.removeEventListener("pointercancel", onUp);
      dom.removeEventListener("pointerleave", onLeave);
      faceGeometry.dispose();
      outlineGeometry.dispose();
      outlineMaterial.dispose();
      for (const g of bevelGeometries) g.dispose();
      for (const c of corners) (c.material as THREE.Material).dispose();
      for (const p of planes) {
        (p.material as THREE.MeshBasicMaterial).dispose();
        (p.userData.hover as THREE.Texture).dispose();
        (p.userData.normal as THREE.Texture).dispose();
      }
      renderer.dispose();
      dom.remove();
    };
  }, [api]);

  return (
    <div
      className="absolute top-4 right-[calc(470px+2rem)] z-10 select-none"
      style={{ width: FRAME_PX, height: FRAME_PX }}
      role="group"
      aria-label="View cube"
    >
      <div ref={mountRef} className="absolute" style={{ left: (FRAME_PX - CANVAS_PX) / 2, top: (FRAME_PX - CANVAS_PX) / 2 }} />
      {BUTTONS.map((b) => (
        <button
          key={b.key}
          className={`btn btn-icon absolute ${b.pos}`}
          style={{ width: BUTTON_PX, height: BUTTON_PX }}
          title={b.title}
          aria-label={b.title}
          onClick={() => {
            const view = api.current;
            if (!view) return;
            if (b.home) view.home();
            else if (b.turn) view.turn(b.turn[0], b.turn[1]);
          }}
        >
          {b.glyph}
        </button>
      ))}
    </div>
  );
}
