import os, json, csv
from pathlib import Path

def collect_visible_frames(json_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = {}
    for p in data.get("points", []):
        fid = int(p["frameIndex"])
        frames.setdefault(fid, []).append(p)

    visible_abs = []
    for fid, pts in frames.items():
        if all(pt.get("vis", True) for pt in pts):
            visible_abs.append(fid)
    return visible_abs  

def generate_visible_json_for_case(training_case_dir, surg_case_ann_dir):
    training_case_dir = Path(training_case_dir)
    surg_case_ann_dir = Path(surg_case_ann_dir)
    mapping_csv = training_case_dir / 'clip_to_video_mapping.csv'

    if not mapping_csv.exists():
        print(f"[WARN] missing mapping file: {mapping_csv}")
        return

    with mapping_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
       
            json_file   = row["json_file"]
            start_frame = int(row["start_frame"])
            end_frame   = int(row["end_frame"])
            out_dir     = Path(row["out_dir"])

            json_path = surg_case_ann_dir / json_file
            if not json_path.exists():
                print(f"[WARN] annotation missing: {json_path}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            visible_abs = collect_visible_frames(json_path)

            visible_local = []
            for fid in visible_abs:
                if start_frame <= fid <= end_frame:
                    local_id = fid - start_frame + 1
                    visible_local.append(local_id)

            visible_local.sort()

            out_json_path = out_dir / "visible_frames.json"
            with out_json_path.open('w') as fout:
                json.dump({"visible_frames": visible_local}, fout, indent=2)

            print(f"[OK] {json_file} -> {out_json_path} ({len(visible_local)} frames)")

def map_case_name(training_case_name: str) -> str:
    if training_case_name.startswith("cholec") and not training_case_name.startswith("cholec80"):
        num = training_case_name.replace("cholec", "")
        return f"cholec80_{num.zfill(2)}"
    return training_case_name

def main():
    TRAIN_ROOT = "/data/home/hao/chenyan/data/training_data"
    SURG_ROOT  = "/data/home/hao/chenyan/ori_data/surg_act_09232025"

    total_cases = 0
    found_cases = 0
    missing_cases = []

    for case_dir in sorted(os.listdir(TRAIN_ROOT)):
        training_case_path = os.path.join(TRAIN_ROOT, case_dir)
        if not os.path.isdir(training_case_path):
            continue

        total_cases += 1
        surg_case_path = os.path.join(SURG_ROOT, map_case_name(case_dir))
        annotation_dir = os.path.join(surg_case_path, "annotation")

        if not os.path.exists(annotation_dir):
            print(f"[WARN] missing annotation dir for {case_dir}")
            missing_cases.append(case_dir)
            continue

        found_cases += 1
        print(f"[CASE] {case_dir} -> {annotation_dir}")
        generate_visible_json_for_case(training_case_path, annotation_dir)

    print("\n========== SUMMARY ==========")
    print(f"Total cases scanned: {total_cases}")
    print(f"Cases found: {found_cases}")
    print(f"Cases missing: {len(missing_cases)}")
    if missing_cases:
        print("Missing list:", ", ".join(missing_cases))


if __name__ == "__main__":
    main()
