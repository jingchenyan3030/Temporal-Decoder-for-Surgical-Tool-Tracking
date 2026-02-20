import argparse
import shutil
from pathlib import Path

ROOT = Path("/data/home/hao/chenyan/data/0923_by_action")

def has_nonempty_images(video_dir: Path) -> bool:
    img_dir = video_dir / "images"
    if not img_dir.is_dir():
        return False

    exts = (".png", ".jpg", ".jpeg")
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", type=str, default=None,
                    help="Only process one action (e.g., grasp/clip/cut/dissect). If omitted, process all.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would be deleted without deleting.")
    args = ap.parse_args()

    if not ROOT.is_dir():
        raise RuntimeError(f"Root not found: {ROOT}")

    print(f"[INFO] Scanning root: {ROOT}")
    if args.action:
        print(f"[INFO] Target action: {args.action}")
    if args.dry_run:
        print("[INFO] DRY RUN (no deletion)")

    deleted = 0
    kept = 0

    for action_dir in sorted(ROOT.iterdir()):
        if not action_dir.is_dir():
            continue
        if args.action and action_dir.name != args.action:
            continue

        for case_dir in sorted(action_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            for video_dir in sorted(case_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                if not video_dir.name.startswith("video_"):
                    continue

                if has_nonempty_images(video_dir):
                    kept += 1
                    continue

    
                rel = video_dir.relative_to(ROOT)
                if args.dry_run:
                    print(f"[DEL] (dry) {rel}")
                else:
                    print(f"[DEL] {rel}")
                    shutil.rmtree(video_dir)
                deleted += 1

    print(f"\n[SUMMARY] kept={kept}, deleted={deleted}")

if __name__ == "__main__":
    main()
