"""Draw runs/phase2/force.svg from the phase 2 logs: force over time for each run, with the limit line.

    python3 tools/plot_phase2.py        (standard library only)
"""
import csv
import json
from pathlib import Path

root = Path(__file__).parent.parent / "runs/phase2"
runs = [("hold", "Arm holds (no yielding)", "#8793A6"), ("yield_gain0.08", "Yields, gain 0.08", "#E0A23B"),
        ("yield_gain0.30", "Yields, gain 0.30", "#3B82C4"), ("yield_gain1.00", "Yields, gain 1.00", "#1B9A66")]
W, H, L, R, T, B = 1200, 620, 80, 30, 60, 70
data, tmax, fmax, cap = [], 0, 0, 1.96
for name, label, col in runs:
    f = root / name / "log.csv"
    if not f.exists():
        continue
    rows = [(float(r["t_s"]), float(r["force_N"]), float(r["finger_pull_m"])) for r in csv.DictReader(f.open())]
    s = json.loads((root / name / "summary.json").read_text())
    cap = s["cap_N"]
    data.append((label + f": peak {s['max_force_N']:.2f} N ({s['max_force_gf']:.0f} gf)", col, rows))
    tmax, fmax = max(tmax, rows[-1][0]), max(fmax, max(r[1] for r in rows))
fmax = max(fmax * 1.08, cap * 1.3)
x = lambda t: L + (W - L - R) * t / tmax
y = lambda v: H - B - (H - T - B) * v / fmax
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="#F4F7FB"/>',
       f'<text x="{L}" y="34" font-size="24" font-weight="700" fill="#0E141F">Force on the gripper while a finger pulls, simulated (ElRobot, Isaac Lab)</text>']
for v in range(0, int(fmax) + 1):
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#D3DCE8"/><text x="{L-12}" y="{y(v)+5:.1f}" font-size="15" text-anchor="end" fill="#465264">{v}</text>')
for t in range(0, int(tmax) + 1, 2):
    out.append(f'<text x="{x(t):.1f}" y="{H-B+24}" font-size="15" text-anchor="middle" fill="#465264">{t}</text>')
out.append(f'<text x="{(L+W-R)/2}" y="{H-18}" font-size="16" text-anchor="middle" fill="#465264">time (s): the finger pulls 8 cm, holds, relaxes, twice</text>')
out.append(f'<text x="22" y="{(T+H-B)/2}" font-size="16" text-anchor="middle" fill="#465264" transform="rotate(-90 22 {(T+H-B)/2})">force (N)</text>')
out.append(f'<line x1="{L}" x2="{W-R}" y1="{y(cap):.1f}" y2="{y(cap):.1f}" stroke="#E63B2E" stroke-width="2" stroke-dasharray="8 6"/><text x="{W-R-6}" y="{y(cap)-8:.1f}" font-size="15" text-anchor="end" fill="#E63B2E">limit {cap} N (200 gf)</text>')
for i, (label, col, rows) in enumerate(data):
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v, _ in rows[::2])
    out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" stroke-linejoin="round"/>')
    out.append(f'<rect x="{L+16}" y="{T+10+i*26}" width="22" height="5" fill="{col}"/><text x="{L+46}" y="{T+18+i*26}" font-size="15" fill="#0E141F">{label}</text>')
out.append("</svg>")
(root / "force.svg").write_text("\n".join(out))
print("wrote", root / "force.svg")
