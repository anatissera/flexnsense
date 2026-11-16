#!/usr/bin/env bash
# Last steps for the final joint run: a one-table film in which every filmed episode succeeds (tries a few sets of
# tables and says which it used), and the recorded episodes as Parquet.
#   bash ~/flexnsense_sim/rl/team_finish.sh <run folder> <output name under ~/flexnsense-sim/runs>
source ~/isaac-env.sh; cd ~/IsaacLab
RUN=~/IsaacLab/logs/skrl/flex_team/$1; R=~/flexnsense-sim/runs/$2
CK="$RUN/checkpoints/$(ls -v "$RUN/checkpoints" | grep '^agent_' | tail -1)"
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python "$@"; }
for seed in 22 23 24 25 26; do
  rm -rf "$R/film_clean"
  run ~/flexnsense_sim/rl/film_team.py --checkpoint "$CK" --episodes 4 --num_envs 8 --seed $seed --out "$R/film_clean" > ~/t_film_clean.log 2>&1
  grep -E "^\[FILM\] episode|^\[READER\]" ~/t_film_clean.log | cut -c1-260 | sed "s/^/tables set $seed: /"
  if [ "$(grep -c 'loop placed: True' ~/t_film_clean.log)" = "4" ]; then echo "USING tables set $seed"; break; fi
done
ffmpeg -y -loglevel error -framerate 10 -i "$R/film_clean/frames/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 19 "$R/two_arms_one_table_clean.mp4"
echo "== dataset"
D=~/flexnsense-sim/deliver_team; mkdir -p "$D/data"
run ~/flexnsense_sim/rl/record_team.py --checkpoint "$CK" --episodes 1024 --out ~/flexnsense-sim/dataset/team > ~/t_rec_team.log 2>&1; grep -E "^\[REC\]|Traceback" -A6 ~/t_rec_team.log | cut -c1-900
~/pq/bin/python ~/flexnsense_sim/rl/rollouts_to_parquet.py ~/flexnsense-sim/dataset/team_pull.npz "$D/data/team_pull_arm.parquet"
~/pq/bin/python ~/flexnsense_sim/rl/rollouts_to_parquet.py ~/flexnsense-sim/dataset/team_move.npz "$D/data/team_move_arm.parquet"
cp ~/flexnsense-sim/dataset/team_report.json "$D/team_report.json"
ls -la "$D/data" "$R"/*.mp4
echo "== FINISH DONE"
