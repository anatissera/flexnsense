#!/usr/bin/env bash
# Record 08_physical_scale.py to an mp4. On the workstation:  bash ~/flexnsense_sim/record.sh <tag> [extra arguments...]
# Frames go to ~/flexnsense-sim/runs/physical/<tag>/, the video to ~/flexnsense-sim/runs/physical/<tag>.mp4
source ~/isaac-env.sh; cd ~/IsaacLab
tag=$1; shift
rm -rf ~/flexnsense-sim/runs/physical/"$tag"
uv run --extra isaacsim python ~/flexnsense_sim/08_physical_scale.py --tag "$tag" "$@" > ~/"rec_$tag.log" 2>&1
ffmpeg -y -loglevel error -framerate 30 -i ~/flexnsense-sim/runs/physical/"$tag"/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ~/flexnsense-sim/runs/physical/"$tag".mp4
grep -E "summary|STOPPED|Traceback" ~/"rec_$tag.log" | cut -c1-300
ls -la ~/flexnsense-sim/runs/physical/"$tag".mp4
