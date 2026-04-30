import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def rotation_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def augment_second_person(person_a: np.ndarray, rotate_deg: float = 35.0, mirror_x: bool = True) -> np.ndarray:
    # person_a: [T, J, 3]
    person_b = person_a.copy()
    if mirror_x:
        person_b[..., 0] *= -1.0
    r = rotation_z(rotate_deg)
    person_b = person_b @ r.T
    person_b[..., 0] += 0.6  # lateral offset to avoid overlap
    return person_b


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_3dpw(root: Path, out_dir: Path, max_files: int) -> dict:
    files = sorted((root / "sequenceFiles").rglob("*.pkl"))
    info = {"dataset": "3DPW", "is_multi_person": True, "files_seen": len(files), "files_written": 0}
    ensure_dir(out_dir)
    for idx, fp in enumerate(files[:max_files]):
        with open(fp, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        joints = data.get("jointPositions", None)
        if not joints or len(joints) < 2:
            continue
        # each person: [T, 72], reshape to [T, 24, 3]
        p1 = np.asarray(joints[0], dtype=np.float32).reshape(-1, 24, 3)
        p2 = np.asarray(joints[1], dtype=np.float32).reshape(-1, 24, 3)
        out_fp = out_dir / f"{fp.stem}.npz"
        np.savez_compressed(out_fp, person_a=p1, person_b=p2, source=str(fp))
        info["files_written"] += 1
    return info


def process_amass(root: Path, out_dir: Path, max_files: int) -> dict:
    files = sorted(root.rglob("*_poses.npz"))
    info = {"dataset": "amass", "is_multi_person": False, "files_seen": len(files), "files_written": 0}
    ensure_dir(out_dir)
    for fp in files[:max_files]:
        z = np.load(fp)
        if "trans" in z.files:
            p1 = z["trans"].astype(np.float32)[:, None, :]  # [T,1,3]
        elif "poses" in z.files:
            # fallback proxy joint from root orientation entries
            p1 = z["poses"].astype(np.float32)[:, :3][:, None, :]
        else:
            continue
        p2 = augment_second_person(p1, rotate_deg=25.0, mirror_x=True)
        out_fp = out_dir / f"{fp.stem}.npz"
        np.savez_compressed(out_fp, person_a=p1, person_b=p2, source=str(fp), synthetic=True)
        info["files_written"] += 1
    return info


def process_h36m(root: Path, out_dir: Path, max_files: int) -> dict:
    files = sorted((root / "dataset").rglob("*.txt"))
    info = {"dataset": "h3.6m", "is_multi_person": False, "files_seen": len(files), "files_written": 0}
    ensure_dir(out_dir)
    for fp in files[:max_files]:
        arr = np.loadtxt(fp, delimiter=",", dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] % 3 != 0:
            continue
        j = arr.shape[1] // 3
        p1 = arr.reshape(arr.shape[0], j, 3)
        p2 = augment_second_person(p1, rotate_deg=40.0, mirror_x=True)
        out_fp = out_dir / f"{fp.parent.name}_{fp.stem}.npz"
        np.savez_compressed(out_fp, person_a=p1, person_b=p2, source=str(fp), synthetic=True)
        info["files_written"] += 1
    return info


def process_mupots(root: Path, out_dir: Path, max_files: int) -> dict:
    annot_files = sorted(root.rglob("annot.mat"))
    info = {"dataset": "MuPots-3d", "is_multi_person": True, "files_seen": len(annot_files), "files_written": 0}
    ensure_dir(out_dir)
    # Keep multi-person source as index for pretrain loader.
    for fp in annot_files[:max_files]:
        mat = loadmat(fp)
        key = "annotations" if "annotations" in mat else None
        if key is None:
            continue
        out_fp = out_dir / f"{fp.parent.name}_index.npz"
        np.savez_compressed(out_fp, source=str(fp), key=key)
        info["files_written"] += 1
    return info


def main():
    parser = argparse.ArgumentParser(description="Check H2H datasets and build two-person augmented data.")
    parser.add_argument("--datasets-root", type=str, default="/data/user/qkh/datasets")
    parser.add_argument("--max-files-per-dataset", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.datasets_root)
    mappings = {
        "3DPW": root / "3DPW",
        "amass": root / "amass",
        "h3.6m": root / "h3.6m",
        "MuPots-3d": root / "MuPots-3d",
    }

    reports = []
    reports.append(process_3dpw(mappings["3DPW"], mappings["3DPW"] / "data_aug", args.max_files_per_dataset))
    reports.append(process_amass(mappings["amass"], mappings["amass"] / "data_aug", args.max_files_per_dataset))
    reports.append(process_h36m(mappings["h3.6m"], mappings["h3.6m"] / "data_aug", args.max_files_per_dataset))
    reports.append(process_mupots(mappings["MuPots-3d"], mappings["MuPots-3d"] / "data_aug", args.max_files_per_dataset))

    summary = {
        "datasets_root": str(root),
        "results": reports,
    }
    out = root / "h2h_pretrain_data_aug_report.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved report -> {out}")


if __name__ == "__main__":
    main()
