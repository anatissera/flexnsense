#!/usr/bin/env bash
# From the trained full-bench pull policy: measure it, check the exported files give the same answers as the trained
# network, assemble the policy folder, run the no-robot self-test, and film it.
#   bash ~/flexnsense_sim/rl/bench_deliver.sh <run folder under ~/IsaacLab/logs/rsl_rl/flex_pull_bench>
source ~/isaac-env.sh; cd ~/IsaacLab
RUN=~/IsaacLab/logs/rsl_rl/flex_pull_bench/$1; CK=$(ls -v "$RUN"/model_*.pt | tail -1); CK0="$RUN/model_0.pt"
R=~/flexnsense-sim/runs/rl_bench; S=~/flexnsense-sim/stage; P="$S/isaac-sim/policies/pull_bench_v1"
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python "$@"; }
rm -rf "$R"; mkdir -p "$R" "$P"
echo "== measure ($CK)"
run ~/flexnsense_sim/rl/eval_pull.py --bench --checkpoint "$CK" --out "$R/eval" > ~/b_eval.log 2>&1; grep -E "^\[EVAL\] (policy|hold)|Traceback" -A3 ~/b_eval.log | cut -c1-420
echo "== policy folder"
cp "$R"/eval/{policy.pt,policy.onnx,policy.onnx.data,summary.json} "$P/"; cp "$CK" "$P/checkpoint.pt"; cp -R "$RUN/params" "$P/"
cat > "$P/flex_sim.json" <<'JSON'
{"kind": "pull", "task": "Flex-PullBench-v0", "rate_hz": 10, "action_scale": 0.06, "cap_n": 1.96, "tip_m": 0.10,
 "start_rad": [0.0, 0.04, 1.057, 0.072, 0.0, -1.169, 0.0], "driven_motors": [1, 2, 3, 4, 6], "obs_dim": 29,
 "trained": "2026-11-15, simulation only (Isaac Lab, 4096 benches, PPO)"}
JSON
echo "== do the exported files answer like the trained network?"
uv run --extra isaacsim python - "$R/eval/io_sample.npz" "$P" <<'PY'
import sys, numpy as np, torch
d = np.load(sys.argv[1]); obs, act = d["obs"].astype(np.float32), d["action"]
m = torch.jit.load(sys.argv[2] + "/policy.pt", map_location="cpu").eval()
with torch.no_grad(): out = m(torch.from_numpy(obs)).numpy()
print(f"TorchScript: largest difference from the trained network on {len(obs)} samples = {np.abs(out - act).max():.2e}")
try:
    import onnxruntime as ort
    s = ort.InferenceSession(sys.argv[2] + "/policy.onnx", providers=["CPUExecutionProvider"])
    o = np.concatenate([s.run(None, {s.get_inputs()[0].name: obs[i : i + 1]})[0] for i in range(len(obs))])  # the file takes one frame at a time
    print(f"ONNX:        largest difference = {np.abs(o - act).max():.2e}")
except ImportError:
    print("ONNX: onnxruntime not installed here, not checked")
PY
echo "== self-test without a robot"
uv run --extra isaacsim python "$S/isaac-sim/policies/selftest_pull.py" --policy "$P" --pulls 40
echo "== videos"
run ~/flexnsense_sim/rl/eval_pull.py --bench --video --checkpoint "$CK" --out "$R/video" > ~/b_video.log 2>&1
ffmpeg -y -loglevel error -framerate 10 -i "$R/video/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 19 "$R/pull_bench_policy_vs_held_still.mp4"
run ~/flexnsense_sim/rl/film_movie.py --bench --before "$CK0" --after "$CK" --out "$R/movie" > ~/b_film.log 2>&1; grep -E "FILM|Traceback" -A4 ~/b_film.log | head -4
ffmpeg -y -loglevel error -framerate 10 -i "$R/movie/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 19 "$R/pull_bench_training_movie_1_to_1024.mp4"
ls -la "$R"/*.mp4 "$P"
