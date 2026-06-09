import copy
from typing import Tuple

import torch
from torch import nn


def _to_joint_repr(x: torch.Tensor, num_joints: int) -> torch.Tensor:
    b, t, c = x.shape
    return x.view(b, t, num_joints, 3).permute(0, 2, 1, 3).contiguous()  # B,J,T,3


def _to_flat_repr(x: torch.Tensor) -> torch.Tensor:
    b, j, t, c = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, t, j * c)


class MotionEncoder(nn.Module):
    """Three-layer MLP encoder for DCT-domain motion tokens."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        # motion: B,J,T,3 -> B,J,T,D
        return self.net(motion)


class GCBlock(nn.Module):
    """Lightweight GCN block with learnable spatial/temporal adjacencies."""

    def __init__(self, channels: int, num_joints: int, seq_len: int, dropout: float):
        super().__init__()
        self.aspatial = nn.Parameter(torch.eye(num_joints))
        self.atemporal = nn.Parameter(torch.eye(seq_len))
        self.bn = nn.BatchNorm2d(channels)
        self.proj = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,J,T
        y = self.bn(x)
        y = torch.einsum("ij,bcjt->bcit", self.aspatial, y)
        y = torch.einsum("tu,bcju->bcjt", self.atemporal, y)
        y = self.proj(y.permute(0, 2, 3, 1))
        y = self.drop(self.relu(y))
        y = y.permute(0, 3, 1, 2).contiguous()
        return y + x


class STCrossAttention(nn.Module):
    """Spatial-then-temporal cross attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.spatial = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q,kv: B,J,T,D
        b, j, t, d = q.shape
        b_kv, j_kv, t_kv, d_kv = kv.shape
        if b_kv != b or t_kv != t or d_kv != d:
            raise ValueError(
                f"q and kv must share batch/time/channel dims, got q={tuple(q.shape)}, kv={tuple(kv.shape)}"
            )

        qs = q.permute(0, 2, 1, 3).reshape(b * t, j, d)
        ks = kv.permute(0, 2, 1, 3).reshape(b * t, j_kv, d)
        spatial, _ = self.spatial(qs, ks, ks)
        spatial = self.norm_s(spatial + qs).reshape(b, t, j, d).permute(0, 2, 1, 3)

        qt = spatial.reshape(b * j, t, d)
        if j_kv == j:
            kt = kv.reshape(b * j, t, d)
        else:
            kt = kv.mean(dim=1, keepdim=True).expand(-1, j, -1, -1).reshape(b * j, t, d)
        temporal, _ = self.temporal(qt, kt, kt)
        temporal = self.norm_t(temporal + qt).reshape(b, j, t, d)
        return temporal


class IFB(nn.Module):
    """Interaction Feature Bridge for stage-2 fine-tuning."""

    def __init__(self, embed_dim: int, num_joints: int, num_heads: int, dropout: float):
        super().__init__()
        self.joint_embed = nn.Parameter(torch.randn(1, num_joints, 1, embed_dim) * 0.02)
        self.spatial = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,J,T,D
        b, j, t, d = x.shape
        x = x + self.joint_embed[:, :j]
        xs = x.permute(0, 2, 1, 3).reshape(b * t, j, d)
        xs, _ = self.spatial(xs, xs, xs)
        xs = xs.reshape(b, t, j, d).permute(0, 2, 1, 3).contiguous()

        xt = xs.permute(0, 1, 3, 2).reshape(b * j, d, t)
        xt = self.temporal(xt).reshape(b, j, d, t).permute(0, 1, 3, 2)
        return self.norm(xt + x)


class JointGatedFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, intra: torch.Tensor, inter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fuse = torch.cat([intra, inter], dim=-1)
        g = self.gate(fuse)  # B,J,T,1
        out = g * intra + (1.0 - g) * inter
        return out, g


class AINet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.human_joint = self.config.motion.dim1 // 3
        self.robot_joint = self.config.motion.dim2 // 3
        self.seq_len = self.config.motion.harper_input_length_dct
        self.pred_len = getattr(self.config.motion, "harper_target_length_train", 10)
        self.embed_dim = getattr(self.config.motion_mlp, "embed_dim", 64)
        self.dropout = getattr(self.config.motion_mlp, "dropout", 0.1)
        self.intra_layers = getattr(self.config.motion_mlp, "intra_layers", 12)
        self.inter_layers = getattr(self.config.motion_mlp, "inter_layers", 9)
        self.heads = getattr(self.config.motion_mlp, "attn_heads", 8)
        if self.embed_dim % self.heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by attn_heads ({self.heads})")

        self.encoder_h = MotionEncoder(self.embed_dim, self.dropout)
        self.encoder_r = MotionEncoder(self.embed_dim, self.dropout)

        self.intra_h = nn.ModuleList(
            [GCBlock(self.embed_dim, self.human_joint, self.seq_len, self.dropout) for _ in range(self.intra_layers)]
        )
        self.intra_r = nn.ModuleList(
            [GCBlock(self.embed_dim, self.robot_joint, self.seq_len, self.dropout) for _ in range(self.intra_layers)]
        )

        self.cross_blocks = nn.ModuleList(
            [STCrossAttention(self.embed_dim, self.heads, self.dropout) for _ in range(self.inter_layers)]
        )
        self.ifb = IFB(self.embed_dim, self.robot_joint, self.heads, self.dropout)
        self.fusion_h = JointGatedFusion(self.embed_dim)
        self.fusion_r = JointGatedFusion(self.embed_dim)

        self.decoder_h = nn.Linear(self.embed_dim, 3)
        self.decoder_r = nn.Linear(self.embed_dim, 3)
        self.rec_h = nn.Linear(self.embed_dim, 3)
        self.rec_r = nn.Linear(self.embed_dim, 3)

        self.stage = 1
        self.set_stage(1)

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad = trainable

    def set_stage(self, stage: int) -> None:
        self.stage = stage
        freeze_backbone = stage == 2
        self._set_trainable(self.intra_h, not freeze_backbone)
        self._set_trainable(self.intra_r, not freeze_backbone)
        self._set_trainable(self.cross_blocks, not freeze_backbone)

    def _run_intra(self, feat: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        # feat: B,J,T,D -> B,D,J,T
        x = feat.permute(0, 3, 1, 2).contiguous()
        for blk in blocks:
            x = blk(x)
        return x.permute(0, 2, 3, 1).contiguous()

    def _run_inter(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = q
        for blk in self.cross_blocks:
            x = blk(x, kv)
        return x

    def _decode_future(self, fused: torch.Tensor, head: nn.Linear, joints: int) -> torch.Tensor:
        # fused: B,J,T,D -> forecast from last token
        last = fused[:, :, -1:, :].expand(-1, -1, self.pred_len, -1)
        pred = head(last)  # B,J,P,3
        pred = pred.permute(0, 2, 1, 3).contiguous().view(fused.shape[0], self.pred_len, joints * 3)
        return pred

    def _decode_recon(self, fused: torch.Tensor, head: nn.Linear, joints: int) -> torch.Tensor:
        rec = head(fused)  # B,J,T,3
        rec = rec.permute(0, 2, 1, 3).contiguous().view(fused.shape[0], self.seq_len, joints * 3)
        return rec

    def forward(self, motion_input1: torch.Tensor, motion_input2: torch.Tensor, nb_iter=0):
        # motion_input: B,T,C
        h_pos = _to_joint_repr(motion_input1, self.human_joint)
        r_pos = _to_joint_repr(motion_input2, self.robot_joint)
        f_en_h = self.encoder_h(h_pos)
        f_en_r = self.encoder_r(r_pos)

        f_intra_h = self._run_intra(f_en_h, self.intra_h)
        f_intra_r = self._run_intra(f_en_r, self.intra_r)

        inter_r_for_h = self.ifb(f_en_r) if self.stage == 2 else f_en_r
        inter_h_for_r = f_en_h
        f_inter_h = self._run_inter(f_en_h, inter_r_for_h)
        f_inter_r = self._run_inter(f_en_r, inter_h_for_r)

        f_fuse_h, g_h = self.fusion_h(f_intra_h, f_inter_h)
        f_fuse_r, g_r = self.fusion_r(f_intra_r, f_inter_r)

        # predict + reconstruct
        pred_h = self._decode_future(f_fuse_h, self.decoder_h, self.human_joint)
        pred_r = self._decode_future(f_fuse_r, self.decoder_r, self.robot_joint)
        rec_h = self._decode_recon(f_fuse_h, self.rec_h, self.human_joint)
        rec_r = self._decode_recon(f_fuse_r, self.rec_r, self.robot_joint)

        # Keep legacy return shape style used by train.py.
        # alpha/alpha2/beta/beta2 are reused as gate diagnostics.
        alpha = g_h.mean(dim=2).permute(0, 2, 1).contiguous()   # B,1,Jh
        alpha2 = 1.0 - alpha
        beta = g_r.mean(dim=2).permute(0, 2, 1).contiguous()    # B,1,Jr
        beta2 = 1.0 - beta

        # stash recon outputs for external loss use when needed
        self.last_recon_h = rec_h
        self.last_recon_r = rec_r
        # Human-motion prediction task: only predict person-1 future.
        # Robot branch is kept as interaction context and for reconstruction.
        return pred_h, alpha, alpha2, beta, beta2
