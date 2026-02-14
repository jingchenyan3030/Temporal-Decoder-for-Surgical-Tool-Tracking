import os
from pathlib import Path
import shutil

from gradio_client import file

KPT_root = Path("/data/home/hao/chenyan/data/0923_training_data")
count = 0
for case in os.listdir(KPT_root):
    case_path = os.path.join(KPT_root, case)
    if not os.path.isdir(case_path):
        continue
    for video in os.listdir(case_path):
        video_path = os.path.join(case_path, video)

        sam_dir = os.path.join(video_path, "sam_results")
        if not os.path.exists(sam_dir):
            continue
        if os.path.isdir(sam_dir):
            shutil.rmtree(sam_dir)
        count += 1
        print(f"Deleted: {sam_dir}")