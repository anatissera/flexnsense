#!/usr/bin/env bash
# The record of the learning: the same table filmed with the arms as they were at several moments of one training
# run, from before any learning to the end, joined into one video.
#   bash ~/flexnsense_sim/rl/film_learning.sh <run folder under ~/IsaacLab/logs/skrl/flex_team> <minutes the run took>
source ~/isaac-env.sh; cd ~/IsaacLab
RUN=~/IsaacLab/logs/skrl/flex_team/$1; MIN=${2:-60}; OUT=${LEARN_OUT:-~/flexnsense-sim/runs/team/learning}; mkdir -p "$OUT"
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_team.py --num_envs 4 --episodes 1 "$@"; }
all=($(ls -v "$RUN"/checkpoints | grep '^agent_' | sed 's/agent_\(.*\)\.pt/\1/'))
last=${all[-1]}
picks=(${all[0]} ${all[$(( ${#all[@]} / 4 ))]} ${all[$(( ${#all[@]} / 2 ))]} $last)
n=0
run --untrained --stage "before any learning" --out "$OUT/s$n" > "$OUT/s$n.log" 2>&1
ffmpeg -y -loglevel error -framerate 10 -i "$OUT/s$n/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/clip_$n.mp4"; echo "file 'clip_$n.mp4'" > "$OUT/list.txt"
grep -E "^\[FILM\] episode" "$OUT/s$n.log" | sed "s/^/before learning: /"
for step in $(printf '%s\n' "${picks[@]}" | awk '!seen[$0]++'); do
  n=$((n + 1)); mins=$(( step * MIN / last ))
  run --checkpoint "$RUN/checkpoints/agent_$step.pt" --stage "after $mins min of learning" --out "$OUT/s$n" > "$OUT/s$n.log" 2>&1
  ffmpeg -y -loglevel error -framerate 10 -i "$OUT/s$n/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/clip_$n.mp4"; echo "file 'clip_$n.mp4'" >> "$OUT/list.txt"
  grep -E "^\[FILM\] episode|^\[READER\]" "$OUT/s$n.log" | cut -c1-260 | sed "s/^/after $mins min: /"
done
ffmpeg -y -loglevel error -f concat -safe 0 -i "$OUT/list.txt" -c copy "$OUT/learning_record.mp4" && ls -la "$OUT/learning_record.mp4"
