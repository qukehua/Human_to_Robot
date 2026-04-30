import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.harper_config import config as base_config
from network.model import AINet


def get_dct_matrix(n: int):
    dct_m = np.eye(n)
    for k in np.arange(n):
        for i in np.arange(n):
            w = np.sqrt(2 / n)
            if k == 0:
                w = np.sqrt(1 / n)
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / n)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def adapt_joint_count(x: np.ndarray, target_joints: int) -> np.ndarray:
    # x: [T, J, 3]
    t, j, c = x.shape
    assert c == 3
    if j == target_joints:
        return x
    if j > target_joints:
        return x[:, :target_joints, :]
    out = np.zeros((t, target_joints, 3), dtype=x.dtype)
    out[:, :j, :] = x
    return out


class H2HPretrainDataset(Dataset):
    def __init__(self, cfg: dict, obs_len: int, pred_len: int, target_joints: int = 21):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.win = obs_len + pred_len
        self.target_joints = target_joints
        self.files = []
        self.index = []

        sources = cfg.get("datasets", {}).get("sources", [])
        for src in sources:
            p = Path(src["path"])
            if not p.exists():
                continue
            self.files.extend(sorted(p.glob("*.npz")))

        # Build a lightweight sliding window index.
        for fp in self.files:
            try:
                z = np.load(fp, allow_pickle=True)
                if "person_a" not in z.files or "person_b" not in z.files:
                    # Skip index-only files (e.g., current MuPots data_aug).
                    continue
                ta = z["person_a"].shape[0]
                tb = z["person_b"].shape[0]
                t = min(ta, tb)
                if t < self.win:
                    continue
                for st in range(0, t - self.win + 1, 1):
                    self.index.append((fp, st))
            except Exception:
                continue

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fp, st = self.index[idx]
        z = np.load(fp, allow_pickle=True)
        a = z["person_a"].astype(np.float32)
        b = z["person_b"].astype(np.float32)

        a = adapt_joint_count(a, self.target_joints)
        b = adapt_joint_count(b, self.target_joints)

        a = a[st : st + self.win]  # [win, J, 3]
        b = b[st : st + self.win]
        data = np.concatenate([a.reshape(self.win, -1), b.reshape(self.win, -1)], axis=-1)  # [win, 2*J*3]

        motion_input = data[: self.obs_len]
        motion_target = data[self.obs_len :]
        return torch.from_numpy(motion_input), torch.from_numpy(motion_target)


def gen_velocity(m):
    return m[:, 1:] - m[:, :-1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/h2h_pretrain_cfg.yml")
    parser.add_argument("--work-dir", type=str, default="./ckpt_h2h_pretrain")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full training by epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    obs_len = int(cfg["sequence"]["obs_len"])
    pred_len = int(cfg["sequence"]["pred_len"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    num_workers = int(cfg["train"]["num_workers"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    lambda_pre = float(cfg["train"]["lambda_pre"])
    lambda_rec = float(cfg["train"]["lambda_rec"])

    dataset = H2HPretrainDataset(cfg, obs_len=obs_len, pred_len=pred_len, target_joints=21)
    if len(dataset) == 0:
        raise RuntimeError("No valid training windows found. Check data_aug files and cfg paths.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    config = copy.deepcopy(base_config)
    config.motion.harper_input_length = obs_len
    config.motion.harper_input_length_dct = obs_len
    config.motion.harper_target_length_train = pred_len
    config.motion.harper_target_length = pred_len
    config.motion.dim1 = 63
    config.motion.dim2 = 63  # human-human pretrain uses symmetric branches

    model = AINet(config).cuda()
    model.set_stage(1)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dct_m, idct_m = get_dct_matrix(obs_len)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    step = 0
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"pretrain epoch {ep+1}/{epochs}")
        for motion_input, motion_target in pbar:
            motion_input = motion_input.cuda()  # [B,T,126]
            motion_target = motion_target.cuda()  # [B,P,126]
            b, t, _ = motion_input.shape
            in_features = 63

            src1 = motion_input[:, :, :in_features]
            src2 = motion_input[:, :, in_features:]
            src1_dct = torch.matmul(dct_m[:, :, :obs_len], src1)
            src2_dct = torch.matmul(dct_m[:, :, :obs_len], src2)

            pred1, _, _, _, _ = model(src1_dct, src2_dct)
            pred1 = pred1[:, :pred_len]

            tgt1 = motion_target[:, :, :in_features]
            loss_pre = torch.mean(torch.norm(pred1 - tgt1, dim=-1))

            rec1 = model.last_recon_h
            rec2 = model.last_recon_r
            src1_xyz = src1.reshape(b, t, 21, 3).reshape(-1, 3)
            src2_xyz = src2.reshape(b, t, 21, 3).reshape(-1, 3)
            rec1_xyz = rec1.reshape(-1, 3)
            rec2_xyz = rec2.reshape(-1, 3)
            loss_rec = torch.mean(torch.norm(rec1_xyz - src1_xyz, dim=-1)) + torch.mean(
                torch.norm(rec2_xyz - src2_xyz, dim=-1)
            )

            total = lambda_pre * loss_pre + lambda_rec * loss_rec
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(
                {
                    "loss": f"{total.item():.4f}",
                    "pre": f"{loss_pre.item():.4f}",
                    "rec": f"{loss_rec.item():.4f}",
                }
            )

            if args.max_steps > 0 and step >= args.max_steps:
                break

        ckpt = Path(args.work_dir) / f"pretrain_stage1_epoch{ep+1}.pth"
        torch.save(model.state_dict(), ckpt)
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_ckpt = Path(args.work_dir) / "pretrain_stage1_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"saved: {final_ckpt}")
    print(f"total steps: {step}")


if __name__ == "__main__":
    main()
