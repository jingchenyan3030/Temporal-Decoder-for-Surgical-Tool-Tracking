from pathlib import Path
import shutil
from natsort import natsorted

src_root = Path("/path/to/original_dataset")
dst_root = Path("/path/to/small_dataset")

actions = ["clip", "grasp", "cut", "dissect"]
keep_per_case = 3   

for action in actions:
    action_src = src_root / action
    action_dst = dst_root / action
    action_dst.mkdir(parents=True, exist_ok=True)

    case_dirs = [d for d in action_src.iterdir() if d.is_dir() and d.name.startswith("case")]
    case_dirs = natsorted(case_dirs, key=str)

    for case_dir in case_dirs:
        videos = [v for v in case_dir.iterdir() if v.is_dir() and v.name.startswith("video")]
        videos = natsorted(videos, key=str)

        selected_videos = videos[:keep_per_case]

        case_dst = action_dst / case_dir.name
        case_dst.mkdir(parents=True, exist_ok=True)

        for video_dir in selected_videos:
            dst_video_dir = case_dst / video_dir.name
            print(f"Copying {video_dir} -> {dst_video_dir}")
            shutil.copytree(video_dir, dst_video_dir, dirs_exist_ok=True)

print("Done.")