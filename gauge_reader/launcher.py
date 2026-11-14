"""A small control panel for the scale reader: pick a camera, calibrate, run, stop.

    python -m gauge_reader.launcher

It starts `python -m gauge_reader calibrate|run` as a child process (the camera view opens in its own
OpenCV window, as from the terminal) and shows that process's output. The known-weight prompt of `calibrate`
(key `w`) is answered from the "Known weight" box here, since there is no terminal to type into.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

REPO = Path(__file__).resolve().parent.parent
CALIB = REPO / "config" / "scale_calib.json"


MODES = [("Stickers (hand-held)", "stickers"), ("Fixed (clamped still)", "fixed")]


def build_cmd(action: str, camera: int | str, record: bool = False, mode: str = "stickers") -> list[str]:
    """Command line for one action ('calibrate', 'recalibrate' or 'run'). `mode` only applies to a new
    calibration; re-tare and run take the mode from the saved calibration file."""
    cmd = [sys.executable, "-u", "-m", "gauge_reader"]
    if action == "recalibrate":
        cmd += ["calibrate", "--resume"]
    else:
        cmd += [action]
    cmd += ["--camera", str(camera)]
    if action == "calibrate":
        cmd += ["--mode", mode]
    if action == "run" and record:
        cmd.append("--record")
    return cmd


def camera_choices() -> list[tuple[int, str]]:
    try:
        from lib.camera import list_cameras
        cams = [(i, n) for i, n in list_cameras() if "desk view" not in n.lower()]
    except Exception:  # noqa: BLE001 - listing is a convenience only
        cams = []
    return cams or [(0, "Camera 0"), (1, "Camera 1"), (2, "Camera 2")]


def calib_summary() -> str:
    if not CALIB.is_file():
        return "Not calibrated yet: press Calibrate."
    try:
        import gauge_reader.calibration as vc
        loader = getattr(vc, "load_calibration", None)
        c = loader(CALIB) if loader is not None else vc.Calibration.load(CALIB)
        when = str(getattr(c, "created_at", "") or "")[:16].replace("T", " ")
        if getattr(c, "mode", "fixed") == "stickers":
            tare = "tared" if getattr(c, "tare_u", None) is not None else "no tare yet"
            what = f"stickers mode · {c.slope_g_per_u:.0f} g per unit u"
        else:
            tare = "tared" if c.tare_t_px is not None else "no tare yet"
            what = f"fixed mode · {c.slope_g_per_px:.2f} g per pixel"
        return f"Calibrated {when} · {what} · {tare} · {len(c.samples)} known weight(s)"
    except Exception as e:  # noqa: BLE001
        return f"Calibration file unreadable: {e}"


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        root.title("Scale Reader")
        root.minsize(600, 520)

        pad = {"padx": 10, "pady": 6}
        top = ttk.Frame(root)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Camera").pack(side="left")
        self.cams = camera_choices()
        self.cam_var = tk.StringVar()
        self.cam_box = ttk.Combobox(top, textvariable=self.cam_var, state="readonly", width=38,
                                    values=[f"{i} · {n}" for i, n in self.cams])
        iphone = next((k for k, (_, n) in enumerate(self.cams) if "iphone" in n.lower()), 0)
        self.cam_box.current(iphone)
        self.cam_box.pack(side="left", padx=8)
        ttk.Button(top, text="Refresh", command=self.refresh_cams).pack(side="left")

        modef = ttk.Frame(root)
        modef.pack(fill="x", **pad)
        ttk.Label(modef, text="Mode").pack(side="left")
        self.mode_var = tk.StringVar(value=MODES[0][1])
        for text, value in MODES:
            ttk.Radiobutton(modef, text=text, value=value, variable=self.mode_var).pack(side="left", padx=(8, 0))
        ttk.Label(modef, text="(used by Calibrate; Run uses the saved calibration's mode)",
                  foreground="#777").pack(side="left", padx=8)

        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.b_cal = ttk.Button(btns, text="1 · Calibrate", command=lambda: self.start("calibrate"))
        self.b_recal = ttk.Button(btns, text="Re-tare / add weights", command=lambda: self.start("recalibrate"))
        self.b_run = ttk.Button(btns, text="2 · Run", command=lambda: self.start("run"))
        self.b_stop = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        for b in (self.b_cal, self.b_recal, self.b_run, self.b_stop):
            b.pack(side="left", padx=(0, 8))
        self.rec_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text="Record scale crop", variable=self.rec_var).pack(side="left")

        kw = ttk.Frame(root)
        kw.pack(fill="x", **pad)
        ttk.Label(kw, text="Known weight (g)").pack(side="left")
        self.w_var = tk.StringVar()
        self.w_entry = ttk.Entry(kw, textvariable=self.w_var, width=8)
        self.w_entry.pack(side="left", padx=8)
        self.w_entry.bind("<Return>", lambda _e: self.send_weight())
        self.b_send = ttk.Button(kw, text="Send", command=self.send_weight, state="disabled")
        self.b_send.pack(side="left")
        ttk.Label(kw, text="press w in the camera window, then send", foreground="#777").pack(side="left", padx=8)

        self.calib_var = tk.StringVar(value=calib_summary())
        ttk.Label(root, textvariable=self.calib_var).pack(anchor="w", **pad)
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(root, textvariable=self.status_var, font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=10)

        help_text = ("Stickers mode: two red dots on the tube, one just before 0 g and one just past 600 g, facing the "
                     "camera, not under your fingers.\n"
                     "Calibrate: SPACE freeze · click the 0 g line, then the 600 g line · the stickers are found "
                     "automatically (else click sticker 1, then sticker 2) · ENTER · z with the scale empty · "
                     "w known weight · s save · q quit.   Fixed mode: same without the stickers.\n"
                     "Run: z tare · m mask · SPACE pause · q quit.")
        ttk.Label(root, text=help_text, wraplength=580, foreground="#555").pack(anchor="w", **pad)

        logf = ttk.Frame(root)
        logf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(logf, height=12, wrap="word", font=("Menlo", 11), state="disabled")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)

    # ------------------------------------------------------------------ actions
    def camera(self) -> int:
        return self.cams[self.cam_box.current()][0]

    def refresh_cams(self):
        self.cams = camera_choices()
        self.cam_box.configure(values=[f"{i} · {n}" for i, n in self.cams])
        self.cam_box.current(0)

    def start(self, action: str):
        if self.proc and self.proc.poll() is None:
            self.write("A session is already running: press Stop first.\n")
            return
        if action == "run" and not CALIB.is_file():
            self.write("No calibration yet: press Calibrate first.\n")
            return
        cmd = build_cmd(action, self.camera(), self.rec_var.get(), self.mode_var.get())
        self.write("$ " + " ".join(cmd[2:]) + "\n")
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        name = {"calibrate": "Calibrating", "recalibrate": "Re-taring", "run": "Running"}[action]
        self.status_var.set(f"{name} on camera {self.camera()}: the camera window opens in a few seconds.")
        self._set_running(True, calibrating=action != "run")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.write("Stopped.\n")

    def send_weight(self):
        g = self.w_var.get().strip()
        if not g or not (self.proc and self.proc.poll() is None and self.proc.stdin):
            return
        try:
            float(g)
        except ValueError:
            self.write(f"Not a number: {g!r}\n")
            return
        self.proc.stdin.write(g + "\n")
        self.proc.stdin.flush()
        self.write(f"> sent {g} g\n")
        self.w_var.set("")

    # ------------------------------------------------------------------ plumbing
    def _pump(self, proc: subprocess.Popen):
        for line in proc.stdout:
            self.lines.put(line)
        self.lines.put(f"[session ended, exit code {proc.wait()}]\n")

    def poll(self):
        while not self.lines.empty():
            self.write(self.lines.get_nowait())
        if self.proc and self.proc.poll() is not None and str(self.b_stop["state"]) != "disabled":
            self._set_running(False)
            self.status_var.set("Idle.")
            self.calib_var.set(calib_summary())
        self.root.after(100, self.poll)

    def _set_running(self, running: bool, calibrating: bool = False):
        for b in (self.b_cal, self.b_recal, self.b_run):
            b.configure(state="disabled" if running else "normal")
        self.b_stop.configure(state="normal" if running else "disabled")
        self.b_send.configure(state="normal" if running and calibrating else "disabled")

    def write(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def on_close(self):
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
