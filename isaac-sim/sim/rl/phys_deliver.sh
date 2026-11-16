#!/usr/bin/env bash
# Final outputs of the physical-contact task from one checkpoint: numbers, film, recorded episodes with pictures.
#   bash ~/flexnsense_sim/rl/phys_deliver.sh <checkpoint> <episodes to record>
source ~/isaac-env.sh; cd ~/IsaacLab
CK=$1; R=~/flexnsense-sim/runs/phys; D=~/flexnsense-sim/deliver_phys
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python "$@"; }
echo "== numbers on 1024 tables"
run ~/flexnsense_sim/rl/eval_phys.py --checkpoint "$CK" --out "$R/eval" > ~/phys_eval.log 2>&1; grep -E "^\[EVAL\]|Traceback" -A6 ~/phys_eval.log | cut -c1-1500
echo "== film"
rm -rf "$R/final"; run ~/flexnsense_sim/rl/film_phys.py --checkpoint "$CK" --episodes 2 --seed 31 --out "$R/final" > ~/phys_final.log 2>&1
grep -E "^\[FILM\]|Traceback" -A6 ~/phys_final.log | cut -c1-300
ffmpeg -y -loglevel error -framerate 10 -i "$R/final/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 20 "$R/final.mp4"; ls -la "$R/final.mp4"
echo "== FILM DONE"
echo "== episodes with pictures"
rm -rf "$R/record"; run ~/flexnsense_sim/rl/film_phys.py --checkpoint "$CK" --record "$2" --seed 77 --out "$R/record" > ~/phys_record.log 2>&1
grep -E "^\[FILM\] wrote|Traceback" -A6 ~/phys_record.log | cut -c1-300; grep -c "whole sequence done: True" ~/phys_record.log
mkdir -p "$D/data"; ~/pq/bin/python ~/flexnsense_sim/rl/phys_to_parquet.py "$R/record/episodes" "$D/data"
mkdir -p "$D/policy"; cp "$R/eval/policy.pt" "$R/eval/policy.onnx"* "$R/eval/summary.json" "$D/policy/" 2>/dev/null; cp "$CK" "$D/policy/checkpoint.pt"
echo "== DELIVER DONE"
