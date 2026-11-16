#!/usr/bin/env bash
# Everything made from one joint training run: the numbers, the exported policies and their check, the no-robot test
# of the pull policy through the real-robot runner, and the three recordings (learning record, one table, 1024 tables).
#   bash ~/flexnsense_sim/rl/team_deliver.sh <run folder under ~/IsaacLab/logs/skrl/flex_team> <minutes the run took> [output name under ~/flexnsense-sim/runs] [policy folder name]
# Nothing from an earlier run is overwritten if a new output name is given.
source ~/isaac-env.sh; cd ~/IsaacLab
RUNNAME=$1; MIN=${2:-60}; RUN=~/IsaacLab/logs/skrl/flex_team/$RUNNAME
CK="$RUN/checkpoints/$(ls -v "$RUN/checkpoints" | grep '^agent_' | tail -1)"
TAG=${3:-team}; R=~/flexnsense-sim/runs/$TAG; S=~/flexnsense-sim/stage; P="$S/isaac-sim/policies/${4:-team_v1}"
export LEARN_OUT="$R/learning"; mkdir -p "$R"
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python "$@"; }
mp4() { ffmpeg -y -loglevel error -framerate 10 -i "$1/frame_%04d.png" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 19 "$2"; }
echo "== checkpoint: $CK"
echo "== numbers on 1024 tables"
run ~/flexnsense_sim/rl/eval_team.py --checkpoint "$CK" --out "$R/eval" > ~/t_eval.log 2>&1; grep -E "^\[EVAL\]|Traceback" -A6 ~/t_eval.log | cut -c1-1500
echo "== exported policies"
rm -rf "$P"; run ~/flexnsense_sim/rl/export_team.py --checkpoint "$CK" --out "$P" > ~/t_export.log 2>&1; grep -E "^\[EXPORT\]|Traceback" -A6 ~/t_export.log | cut -c1-300
cp "$R/eval/summary.json" "$P/summary.json"; cp "$CK" "$P/checkpoint.pt"; cp -R "$RUN/params" "$P/" 2>/dev/null
cat > "$P/pull/flex_sim.json" <<'JSON'
{"kind": "pull", "task": "Flex-Team-v0", "rate_hz": 10, "action_scale": 0.06, "cap_n": 1.96, "tip_m": 0.10,
 "start_rad": [0.0, 0.04, 1.057, 0.072, 0.0, -1.169, 0.0], "driven_motors": [1, 2, 3, 4, 6], "obs_dim": 34,
 "trained": "2026-11-16, simulation only (Isaac Lab, 4096 tables, two agents, IPPO), together with the move arm"}
JSON
echo "== the pull policy through the real-robot runner, against a stand-in arm and scale"
uv run --extra isaacsim python "$S/isaac-sim/policies/selftest_pull.py" --policy "$P/pull" --pulls 40 2>&1 | tail -4
echo "== recordings"
bash ~/flexnsense_sim/rl/film_learning.sh "$RUNNAME" "$MIN" 2>&1 | tail -12
rm -rf "$R/film"; run ~/flexnsense_sim/rl/film_team.py --checkpoint "$CK" --episodes 4 --num_envs 8 --out "$R/film" > ~/t_film.log 2>&1; grep -E "^\[FILM\]|^\[READER\]|Traceback" -A6 ~/t_film.log | cut -c1-330
mp4 "$R/film/frames" "$R/two_arms_one_table.mp4"
rm -rf "$R/movie"; run ~/flexnsense_sim/rl/film_team_movie.py --checkpoint "$CK" --out "$R/movie" > ~/t_movie.log 2>&1; grep -E "^\[FILM\]|Traceback" -A6 ~/t_movie.log | cut -c1-300
mp4 "$R/movie/frames" "$R/two_arms_1_to_1024_tables.mp4"
ls -la "$R"/*.mp4 "$R/learning/learning_record.mp4"
echo "== DELIVER DONE"
