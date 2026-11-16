#!/usr/bin/env bash
# Everything that is made from the two trained policies: videos, the films, recorded episodes as Parquet, short clips.
#   bash ~/flexnsense_sim/rl/make_deliverables.sh <pull checkpoint> <pull model_0> <move checkpoint> <move model_0>
source ~/isaac-env.sh; cd ~/IsaacLab
PULL=$1; PULL0=$2; MOVE=$3; MOVE0=$4
run() { PYTHONPATH=~/flexnsense_sim/rl uv run --extra isaacsim python "$@"; }
R=~/flexnsense-sim/runs; D=~/flexnsense-sim/deliver; rm -rf "$D"; mkdir -p "$D"/{media,dataset,policies/pull_v1,policies/move_v1}
mp4() { ffmpeg -y -loglevel error -framerate 30 -i "$1/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 19 "$2"; }
gif() { ffmpeg -y -loglevel error -ss "$2" -t "$3" -i "$1" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" "$4"; }

echo "== move: measure, film one arm"
run ~/flexnsense_sim/rl/eval_move.py --checkpoint "$MOVE" --out "$R/rl_move/eval_final" > ~/d_eval_move.log 2>&1; grep -E "^\[EVAL\] \{" ~/d_eval_move.log
run ~/flexnsense_sim/rl/eval_move.py --checkpoint "$MOVE" --video --out "$R/rl_move/video_final" > ~/d_video_move.log 2>&1
mp4 "$R/rl_move/video_final/frames" "$D/media/move_policy_one_arm.mp4"
echo "== move: the film"
rm -rf "$R/rl_move/movie"; run ~/flexnsense_sim/rl/film_move_movie.py --before "$MOVE0" --after "$MOVE" > ~/d_film_move.log 2>&1; grep -E "FILM|Traceback" ~/d_film_move.log
mp4 "$R/rl_move/movie/frames" "$D/media/move_training_movie_1_to_1024_arms.mp4"
echo "== pull: existing videos"
cp "$R/rl_pull/movie/pull_training_movie.mp4" "$D/media/pull_training_movie_1_to_1024_arms.mp4"
cp "$R/rl_pull/video/pull_policy.mp4" "$D/media/pull_policy_trained_vs_held_still.mp4"
cp "$R/rl_pull/training_film/pull_training_progress.mp4" "$D/media/pull_policy_training_progress.mp4"
echo "== recorded episodes"
run ~/flexnsense_sim/rl/record_rollouts.py --task pull --checkpoint "$PULL" --episodes 512 --out ~/flexnsense-sim/dataset/pull.npz > ~/d_rec_pull.log 2>&1; grep -E "REC|Traceback" -A3 ~/d_rec_pull.log | head -4
run ~/flexnsense_sim/rl/record_rollouts.py --task move --checkpoint "$MOVE" --episodes 512 --out ~/flexnsense-sim/dataset/move.npz > ~/d_rec_move.log 2>&1; grep -E "REC|Traceback" -A3 ~/d_rec_move.log | head -4
mkdir -p "$D/dataset/data"
~/pq/bin/python ~/flexnsense_sim/rl/rollouts_to_parquet.py ~/flexnsense-sim/dataset/pull.npz "$D/dataset/data/pull_arm.parquet"
~/pq/bin/python ~/flexnsense_sim/rl/rollouts_to_parquet.py ~/flexnsense-sim/dataset/move.npz "$D/dataset/data/move_arm.parquet"
echo "== policies"
cp "$R"/rl_pull/eval/{policy.onnx,policy.onnx.data,policy.pt,summary.json,trace.csv} "$D/policies/pull_v1/" 2>/dev/null
cp "$PULL" "$D/policies/pull_v1/checkpoint.pt"; cp -R "$(dirname "$PULL")/params" "$D/policies/pull_v1/"
cp "$R"/rl_move/eval_final/{policy.onnx,policy.onnx.data,policy.pt,summary.json} "$D/policies/move_v1/" 2>/dev/null
cp "$MOVE" "$D/policies/move_v1/checkpoint.pt"; cp -R "$(dirname "$MOVE")/params" "$D/policies/move_v1/"
echo "== short clips"
gif "$D/media/pull_training_movie_1_to_1024_arms.mp4" 25.5 4 "$D/media/pull_1024_arms.gif"
gif "$D/media/pull_policy_trained_vs_held_still.mp4" 2.5 4.5 "$D/media/pull_policy.gif"
gif "$D/media/move_training_movie_1_to_1024_arms.mp4" 17.5 4.5 "$D/media/move_policy.gif"
gif "$D/media/move_training_movie_1_to_1024_arms.mp4" 27 4 "$D/media/move_1024_arms.gif"
du -sh "$D"/*; ls -la "$D/media" "$D/dataset/data"
