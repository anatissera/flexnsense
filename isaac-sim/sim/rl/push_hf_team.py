"""Add the two-arm results to the two public Hugging Face repos (run on the workstation; token from ~/.hf_token).

    ~/pq/bin/python ~/flexnsense_sim/rl/push_hf_team.py

Dataset repo lerobot-2026:            two_arms/team_pull_arm.parquet, team_move_arm.parquet, team_report.json, README
Model repo flexnsense-sim-policies:   policies/team_v3/ (simulation only, see its NOTES.md)
Nothing already in the repos is removed.
"""

import argparse
import os

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--model_repo", default="flexnsense-sim-policies")
parser.add_argument("--dataset_repo", default="lerobot-2026")
args = parser.parse_args()

token = open(os.path.expanduser("~/.hf_token")).read().strip()
api = HfApi(token=token)
user = api.whoami()["name"]
home = os.path.expanduser("~/flexnsense-sim")
msg = "Two arms trained together (simulation, hackathon use only)"
api.upload_folder(repo_id=f"{user}/{args.dataset_repo}", repo_type="dataset", folder_path=f"{home}/deliver_team/data",
                  path_in_repo="two_arms", commit_message=msg)
for name in ("team_report.json", "README.md"):
    api.upload_file(repo_id=f"{user}/{args.dataset_repo}", repo_type="dataset", path_or_fileobj=f"{home}/deliver_team/{name}",
                    path_in_repo=f"two_arms/{name}", commit_message=msg)
api.upload_folder(repo_id=f"{user}/{args.model_repo}", repo_type="model", folder_path=f"{home}/stage/isaac-sim/policies/team_v3",
                  path_in_repo="policies/team_v3", commit_message=msg, ignore_patterns=["checkpoint.pt", "*.npz"])
print(f"uploaded to https://huggingface.co/datasets/{user}/{args.dataset_repo}/tree/main/two_arms")
print(f"uploaded to https://huggingface.co/{user}/{args.model_repo}/tree/main/policies/team_v3")
