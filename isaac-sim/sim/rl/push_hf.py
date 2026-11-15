"""Upload the simulation results to Hugging Face, from the workstation: a PUBLIC model repo and a PUBLIC dataset repo.

The token is read from ~/.hf_token on the workstation (one line, created by you; this script never prints it).

    ~/pq/bin/python ~/flexnsense_sim/rl/push_hf.py
    ~/pq/bin/python ~/flexnsense_sim/rl/push_hf.py --model_repo flexnsense-sim-policies --dataset_repo lerobot-2026 --private

Uploads from ~/flexnsense-sim/deliver/ (made by make_deliverables.sh):
  model repo:   policies/, media/, code/ (a copy of ~/flexnsense_sim/rl), README.md from hf_cards/model_README.md
  dataset repo: data/*.parquet, README.md from hf_cards/dataset_README.md
"""

import argparse
import os
import shutil
import sys

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--model_repo", default="flexnsense-sim-policies")
parser.add_argument("--dataset_repo", default="lerobot-2026")
parser.add_argument("--private", action="store_true", help="create the repos private (default: public)")
args = parser.parse_args()

token_file = os.path.expanduser("~/.hf_token")
if not os.path.exists(token_file):
    sys.exit("no ~/.hf_token on this machine: create it first (one line with the token, chmod 600)")
api = HfApi(token=open(token_file).read().strip())
user = api.whoami()["name"]
deliver = os.path.expanduser("~/flexnsense-sim/deliver")
here = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(deliver, "_model")
shutil.rmtree(model_dir, ignore_errors=True)
os.makedirs(model_dir)
for part in ("policies", "media"):
    shutil.copytree(os.path.join(deliver, part), os.path.join(model_dir, part))
shutil.copytree(here, os.path.join(model_dir, "code"), ignore=shutil.ignore_patterns("__pycache__", "hf_cards"))
shutil.copy(os.path.join(here, "hf_cards", "model_README.md"), os.path.join(model_dir, "README.md"))

data_dir = os.path.join(deliver, "dataset")
card = open(os.path.join(here, "hf_cards", "dataset_README.md")).read().replace("<this repo>", f"{user}/{args.dataset_repo}")
open(os.path.join(data_dir, "README.md"), "w").write(card)

for repo, kind, folder, msg in (
    (f"{user}/{args.model_repo}", "model", model_dir, "Simulation policies for the pull arm and the move arm"),
    (f"{user}/{args.dataset_repo}", "dataset", data_dir, "Simulated episodes of the two trained policies"),
):
    api.create_repo(repo, repo_type=kind, private=args.private, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type=kind, folder_path=folder, commit_message=msg)
    where = "https://huggingface.co/" + ("datasets/" if kind == "dataset" else "") + repo
    print("uploaded:", where, "(private)" if args.private else "(public)")
