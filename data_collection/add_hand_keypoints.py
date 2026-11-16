#!/usr/bin/env python
"""Add hand keypoints (WiLoR, overhead camera) to episodes that are already on the Hub.

For every data/<task>/episode_NNNN.parquet in the dataset repo: download it, run WiLoR on the
overhead camera of every frame, add the `hand_*` columns (see hand_keypoints.py) and upload the
file back to the same path. Every other column is written back unchanged. Local copies are
deleted after each upload. The previous version of each file stays in the Hub's git history.

Usage:
    python add_hand_keypoints.py                 # every episode that has no hand columns yet
    python add_hand_keypoints.py --force         # recompute all of them
    python add_hand_keypoints.py --dry-run       # process and report, upload nothing
"""
import argparse
import io
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from hand_keypoints import (FINGERTIPS, HAND_COLUMNS, HandKeypointTracker, find_overhead_camera,
                            rerun_without_user_site, with_hand_columns)


def download(repo: str, path: str, token: str, dst: Path) -> None:
    # Plain HTTP on purpose: `hf download` hung on the laptop this was written on.
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def save_preview(table, hand_camera: int, dst: Path) -> None:
    """Middle frame of the episode, overhead camera, with the keypoints drawn on it."""
    from PIL import Image, ImageDraw

    i = table.num_rows // 2
    jpeg = table.column("images")[i].as_py()[hand_camera]["jpeg"]
    kp = table.column("hand_keypoints_2d")[i].as_py()
    image = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((672, 672), Image.BICUBIC)
    draw = ImageDraw.Draw(image)
    for k in range(21):
        x, y = kp[2 * k] * 672, kp[2 * k + 1] * 672
        r, color = (6, (255, 170, 0)) if k in FINGERTIPS else (3, (0, 230, 255))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    image.save(dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="JustoDario/flexnsense_skill1")
    p.add_argument("--hand-camera", default="auto",
                   help="Index of the overhead camera in `images`, or 'auto' (default): found per episode")
    p.add_argument("--wilor-root", help="WiLoR checkout (default: WILOR_ROOT, else <repo>/WiLoR)")
    p.add_argument("--batch", type=int, default=10, help="Episodes per Hub commit")
    p.add_argument("--only", help="Regex on the file path, to process a subset")
    p.add_argument("--force", action="store_true", help="Recompute episodes that already have hand columns")
    p.add_argument("--dry-run", action="store_true", help="Do not upload")
    p.add_argument("--preview-dir", default="hand_preview", help="Where to write one overlay image per episode")
    args = p.parse_args()
    rerun_without_user_site()

    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, HfApi, get_token

    api, token = HfApi(), get_token()
    pattern = re.compile(r"^data/[^/]+/episode_\d+\.parquet$")
    paths = sorted(f for f in api.list_repo_files(args.repo, repo_type="dataset") if pattern.match(f))
    if args.only:
        paths = [f for f in paths if re.search(args.only, f)]
    print(f"{len(paths)} episodes in {args.repo}")

    tracker = HandKeypointTracker(args.wilor_root)
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hand_kp_", dir=Path(__file__).resolve().parent))
    batch: list[tuple[str, Path]] = []
    total_frames = total_found = 0

    def flush() -> None:
        if batch and not args.dry_run:
            ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local)) for path, local in batch]
            api.create_commit(repo_id=args.repo, repo_type="dataset", operations=ops,
                              commit_message=f"Add hand keypoints (WiLoR, overhead camera) to {len(batch)} episodes")
            print(f"[hub] uploaded {len(batch)} episodes")
        for _path, local in batch:
            local.unlink()
        batch.clear()

    try:
        for n, path in enumerate(paths, 1):
            local = work / path.replace("/", "__")
            download(args.repo, path, token, local)
            table = pq.read_table(local)
            if not args.force and all(c in table.column_names for c in HAND_COLUMNS):
                print(f"[{n}/{len(paths)}] {path}: already has hand columns, skipped")
                local.unlink()
                continue
            t0 = time.time()
            tracker.reset()
            rows = table.column("images").to_pylist()
            if args.hand_camera == "auto":
                cam = find_overhead_camera([c["jpeg"] for c in rows[len(rows) // 2]])
            else:
                cam = int(args.hand_camera)
            detections = [tracker.detect(row[cam]["jpeg"]) for row in rows]
            del rows
            found = sum(d is not None for d in detections)
            table = with_hand_columns(table, detections, cam)
            pq.write_table(table, local)
            save_preview(table, cam, preview_dir / (Path(path).stem + ".jpg"))
            total_frames += len(detections)
            total_found += found
            flag = "" if found == len(detections) else "   <-- frames with no hand"
            print(f"[{n}/{len(paths)}] {path}: overhead camera = images[{cam}], hand in {found}/{len(detections)} frames, "
                  f"{time.time() - t0:.1f} s{flag}", flush=True)
            batch.append((path, local))
            if len(batch) >= args.batch:
                flush()
        flush()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if total_frames:
        print(f"Done. Hand found in {total_found}/{total_frames} frames ({100 * total_found / total_frames:.1f} %). "
              f"Overlay previews in {preview_dir}/")
    if args.dry_run:
        print("Dry run: nothing was uploaded.")


if __name__ == "__main__":
    sys.exit(main())
