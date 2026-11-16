#!/usr/bin/env bash
# The learning record of the physical-contact task: one episode per saved stage, joined into one film.
#   bash ~/flexnsense_sim/rl/phys_learning.sh <run1 folder> <run2 folder>
source ~/isaac-env.sh; cd ~/IsaacLab
L=~/IsaacLab/logs/rsl_rl/flex_phys; R=~/flexnsense-sim/runs/phys/learning; rm -rf "$R"; mkdir -p "$R"
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python ~/flexnsense_sim/rl/film_phys.py "$@"; }
k=0
film() { # checkpoint, label, extra
  run --checkpoint "$1" --episodes 1 --stage "$2" --out "$R/s$k" $3 > ~/phys_learn_$k.log 2>&1
  grep -E "^\[FILM\] episode" ~/phys_learn_$k.log | sed "s/^/$2: /"
  ffmpeg -y -loglevel error -framerate 10 -i "$R/s$k/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 20 "$R/s$k.mp4"
  echo "file '$R/s$k.mp4'" >> "$R/list.txt"; k=$((k+1))
}
film "$L/$1/model_0.pt" "before learning" "--untrained"
film "$L/$1/model_200.pt" "after 11 min"
film "$L/$1/model_400.pt" "after 23 min: reaches, cannot grip yet"
film "$L/$2/model_500.pt" "after 29 min: first grips"
film "$L/$2/model_600.pt" "after 34 min"
film "$L/$2/model_900.pt" "after 51 min: the whole sequence"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$R/list.txt" -c copy "$R/learning_record.mp4"; ls -la "$R/learning_record.mp4"
echo "== LEARNING DONE"
