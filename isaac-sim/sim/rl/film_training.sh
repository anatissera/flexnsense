#!/usr/bin/env bash
# Film the pull policy at several moments of its training and join the clips into one video.
#   bash ~/flexnsense_sim/rl/film_training.sh <run folder under ~/IsaacLab/logs/rsl_rl/flex_pull> [number of arms in view, e.g. 64]
source ~/isaac-env.sh; cd ~/IsaacLab
run=$1; grid=${2:-0}; out=~/flexnsense-sim/runs/rl_pull/training_film${2:+_grid}; rm -rf "$out"; mkdir -p "$out"
n=0
for it in 0 100 200 599; do
  PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/eval_pull.py --video --only_policy --grid "$grid" \
    --checkpoint ~/IsaacLab/logs/rsl_rl/flex_pull/"$run"/model_$it.pt --out "$out/it_$it" \
    --title "Pull policy after $it training iterations ($((it * 24 * 4096 / 1000000)) million steps)" > "$out/it_$it.log" 2>&1
  ffmpeg -y -loglevel error -framerate 30 -i "$out/it_$it/frames/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 "$out/clip_$n.mp4"
  echo "file 'clip_$n.mp4'" >> "$out/list.txt"; n=$((n + 1))
  grep -E "^\[EVAL\] policy" "$out/it_$it.log" | cut -c1-260
done
ffmpeg -y -loglevel error -f concat -safe 0 -i "$out/list.txt" -c copy "$out/pull_training_progress.mp4"
ls -la "$out/pull_training_progress.mp4"
