import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    obj_edge_vectors,
    encode_box_info,
    encode_orientedbox_info,
)


# --------------------------------------------------------------------------------------
# GaussGate
# --------------------------------------------------------------------------------------
class GaussGate(nn.Module):
    """
    Rotated anisotropic Gaussian relation confidence on edges specified by rel_pair_inds.
    - Input proposal.bbox: (N,5) = (cx, cy, w, h, angle[rad]), proposal.size=(W,H).
    - OffsetHead predicts offsets from geometry and predicted semantics only.
    - Output: list[Tensor(N_i,N_i)] with nonzero values only at rel_pair_inds.
    """

    def __init__(
        self,
        hidden_dim: int = 512,

        # Bandwidth and range parameter domain.
        lam_w_init: float = 2.0,
        lam_h_init: float = 2.0,
        lam_min: float = 0.05,
        lam_max: float = 5.0,

        # Initial gamma value decoded with softplus and a lower bound.
        gamma_init: float = 1.5,  # Added in log space before softplus.
        gamma_min: float = 0.3,

        # Other kernel options.
        lam_mode: str = "normalized",  # or "pixels"
        eps: float = 1e-8,
        use_rational_kernel: bool = True,  # True: 1/(1+z); False: exp(-z)
        z_cap: float = 40.0,
        d2_scale: float = 0.5,  # Adjustable version of the legacy 0.25 factor.
    ):
        super().__init__()
        assert lam_mode in ("normalized", "pixels")
        self.eps = float(eps)
        self.lam_mode = lam_mode
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.use_rational_kernel = bool(use_rational_kernel)
        self.z_cap = float(z_cap)
        self.d2_scale = float(d2_scale)

        # Lower bound for gamma.
        self.gamma_min = float(gamma_min)

        # Add learnable offsets to fixed global initial values.
        def _inv_sigmoid_map(x: float, lo: float, hi: float) -> float:
            frac = (x - lo) / (hi - lo + 1e-12)
            frac = min(max(frac, 1e-6), 1.0 - 1e-6)
            return math.log(frac / (1.0 - frac))

        raw_lam_w0 = _inv_sigmoid_map(lam_w_init, self.lam_min, self.lam_max)
        raw_lam_h0 = _inv_sigmoid_map(lam_h_init, self.lam_min, self.lam_max)
        log_gamma0 = math.log(max(float(gamma_init), 1e-6))

        self.register_buffer("raw_lam_w0", torch.tensor(raw_lam_w0, dtype=torch.float32))
        self.register_buffer("raw_lam_h0", torch.tensor(raw_lam_h0, dtype=torch.float32))
        self.register_buffer("log_gamma0", torch.tensor(log_gamma0, dtype=torch.float32))

        # Offset head based on geometry and predicted semantics.
        self.offset_head = OffsetHead(hidden_dim=hidden_dim)

    # Decode constrained parameters.
    def _decode_lam(self, raw: torch.Tensor) -> torch.Tensor:
        return self.lam_min + torch.sigmoid(raw) * (self.lam_max - self.lam_min)

    def _decode_gamma(self, log_gamma: torch.Tensor) -> torch.Tensor:
        # Softplus keeps gamma positive, and the lower bound prevents collapse.
        return self.gamma_min + F.softplus(log_gamma)

    def _normalize_boxes(self, proposal):
        if not hasattr(proposal, "size") or proposal.size is None:
            raise RuntimeError("GaussGate requires proposal.size to be set for normalization.")
        img_w, img_h = proposal.size
        boxes = proposal.bbox.float()
        scale = boxes.new_tensor([img_w, img_h, img_w, img_h, 1.0])
        boxes = boxes / scale
        return boxes, float(img_w), float(img_h)

    @staticmethod
    def _soft_cap(z: torch.Tensor, cap: float) -> torch.Tensor:
        return cap - F.softplus(cap - z)

    def forward(self, entities_proposals, rel_pair_inds):
        """Return one (N_i,N_i) relation confidence matrix per image."""
        relness_matrix = []

        # Predict offsets from geometry and predicted semantics only.
        offsets_list = self.offset_head(entities_proposals, rel_pair_inds)
        # offsets_list: List[Tensor(N_i, 3)]

        # Build a gated relation matrix for each image.
        for proposal, pair_idx, offsets in zip(entities_proposals, rel_pair_inds, offsets_list):
            device = proposal.bbox.device
            dtype = proposal.bbox.dtype
            N = len(proposal)

            pred_rel_matrix = torch.zeros((N, N), device=device, dtype=dtype)
            if N == 0 or pair_idx is None or pair_idx.numel() == 0:
                relness_matrix.append(pred_rel_matrix)
                continue

            if getattr(proposal, "mode", "xywha") != "xywha":
                raise NotImplementedError("GaussGate supports 'xywha' proposal.mode only.")

            # Remove self loops and duplicate directed pairs.
            with torch.no_grad():
                pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
                if pair_idx.numel() == 0:
                    relness_matrix.append(pred_rel_matrix)
                    continue
                pair_idx = torch.unique(pair_idx, dim=0)

            # Add offsets and decode constrained parameters.
            d_raw_lam_w = offsets[:, 0].to(device=device, dtype=dtype)
            d_raw_lam_h = offsets[:, 1].to(device=device, dtype=dtype)
            d_log_gamma = offsets[:, 2].to(device=device, dtype=dtype)

            raw_lam_w = d_raw_lam_w + self.raw_lam_w0.to(device=device, dtype=dtype)
            raw_lam_h = d_raw_lam_h + self.raw_lam_h0.to(device=device, dtype=dtype)
            log_gamma = d_log_gamma + self.log_gamma0.to(device=device, dtype=dtype)

            lam_w_i = self._decode_lam(raw_lam_w)      # (N,)
            lam_h_i = self._decode_lam(raw_lam_h)      # (N,)
            gamma_i = self._decode_gamma(log_gamma)    # (N,)

            # Normalize boxes.
            boxes, img_w, img_h = self._normalize_boxes(proposal)
            boxes = boxes.to(device=device, dtype=dtype)
            cx, cy, w, h, a = boxes.unbind(dim=1)

            # Pixel or normalized coordinate mode.
            if self.lam_mode == "pixels":
                lamw_eff = lam_w_i / img_w
                lamh_eff = lam_h_i / img_h
            else:
                lamw_eff = lam_w_i
                lamh_eff = lam_h_i

            # Per-node anisotropic covariance components.
            c = torch.cos(a)
            s = torch.sin(a)
            edge_w = w
            edge_h = h
            sx = (lamw_eff * edge_w).pow(2) + self.eps
            sy = (lamh_eff * edge_h).pow(2) + self.eps
            S_xx = c * c * sx + s * s * sy
            S_yy = s * s * sx + c * c * sy
            S_xy = c * s * (sx - sy)

            # Edge indices.
            i_idx = pair_idx[:, 0].long().to(device)
            j_idx = pair_idx[:, 1].long().to(device)

            # Pairwise covariance: Sigma_i + Sigma_j.
            Sxx = S_xx[i_idx] + S_xx[j_idx]
            Syy = S_yy[i_idx] + S_yy[j_idx]
            Sxy = S_xy[i_idx] + S_xy[j_idx]

            det = (Sxx * Syy - Sxy * Sxy).clamp_min(self.eps)
            inv11 = Syy / det
            inv12 = -Sxy / det
            inv22 = Sxx / det

            dx = cx[i_idx] - cx[j_idx]
            dy = cy[i_idx] - cy[j_idx]

            quad = inv11 * dx * dx + 2.0 * inv12 * dx * dy + inv22 * dy * dy
            d2 = self.d2_scale * quad.clamp_min(0.0)

            # Use raw d2 without per-image median normalization.
            d2_norm = d2

            # Symmetric gamma pair from the geometric mean.
            gamma_pair = (gamma_i[i_idx] * gamma_i[j_idx]).sqrt()

            # Soft cap followed by the selected kernel.
            z = gamma_pair * d2_norm
            z = self._soft_cap(z, self.z_cap)

            conf_edges = (1.0 / (1.0 + z)) if self.use_rational_kernel else torch.exp(-z)

            pred_rel_matrix[i_idx, j_idx] = conf_edges
            pred_rel_matrix[j_idx, i_idx] = conf_edges

            relness_matrix.append(pred_rel_matrix)

        return relness_matrix


# --------------------------------------------------------------------------------------
# OffsetHead uses geometry and predicted semantics only.
# --------------------------------------------------------------------------------------
class OffsetHead(nn.Module):
    """
    Predict undecoded influence-parameter offsets for each object proposal:
        offsets[i] = [Delta raw_lam_w, Delta raw_lam_h, Delta log_gamma]
    Inputs:
      - proposal.get_field("predict_logits") : (N, num_obj_classes)
      - encode_box_info / encode_orientedbox_info -> (N, 9)
    Returns: List[Tensor(N_i, 3)]
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.cfg = cfg

        # Dimension settings.
        self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.geometry_feat_dim = 128
        self.hidden_dim = int(hidden_dim)

        # Semantic word vectors from GloVe.
        statistics = get_dataset_statistics(cfg)
        obj_classes = statistics["obj_classes"]
        obj_embed_vecs = obj_edge_vectors(
            obj_classes,
            wv_dir=self.cfg.GLOVE_DIR,
            wv_dim=self.embed_dim
        )
        self.obj_sem_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_sem_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        # Geometry encoding.
        self.obj_pos_embed = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.geometry_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.geometry_feat_dim, self.geometry_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Project symbolic geometry and semantics into the hidden space.
        self.symb_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim + self.geometry_feat_dim),
            nn.Linear(self.embed_dim + self.geometry_feat_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Lightweight context block with a gated residual to reduce over-smoothing.
        self.attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=0.1, bias=True
        )
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Gated residual strength.
        self.drop = nn.Dropout(0.1)

        # Offset regression head: [Delta raw_lam_w, Delta raw_lam_h, Delta log_gamma].
        self.param_offset_fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3),
        )
        # Small random initialization avoids a fully symmetric start.
        nn.init.normal_(self.param_offset_fc[-1].weight, std=1e-3)
        nn.init.zeros_(self.param_offset_fc[-1].bias)

    def _encode_geometry(self, proposal) -> torch.Tensor:
        if getattr(proposal, "mode", "xywha") == "xywha":
            pos = encode_orientedbox_info([proposal])
        else:
            pos = encode_box_info([proposal])
        return self.obj_pos_embed(pos)

    def _encode_semantic(self, proposal) -> torch.Tensor:
        # Use detached predicted logits to build a soft semantic embedding.
        pred_logits = proposal.get_field("predict_logits").detach()  # (N, num_obj_classes)
        probs = F.softmax(pred_logits, dim=1)
        sem = probs @ self.obj_sem_embed.weight  # (N, embed_dim)
        return sem

    def _transformer_block(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)               # (L=N, N=1, E=hidden)
        attn_out, _ = self.attn(x, x, x) # Same shape as input: (L, 1, E).
        x = self.ln1(x + self.drop(self.alpha * attn_out))
        ffn_out = self.ffn(x)            # (L, 1, E)
        x = self.ln2(x + self.drop(ffn_out))
        return x.squeeze(1)              # (N, hidden)

    @torch.no_grad()
    def _ensure_device_dtype(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t.to(device=ref.device, dtype=ref.dtype)

    def forward(self, entities_proposals, rel_pair_inds=None):
        """Return offsets_per_image: List[Tensor(N_i, 3)]."""
        offsets_per_image = []

        for proposal in entities_proposals:
            device = proposal.bbox.device
            dtype = torch.float32  # The regression head uses float32 internally.

            N = len(proposal)
            if N == 0:
                offsets_per_image.append(torch.zeros((0, 3), device=device, dtype=dtype))
                continue

            # Encode geometry and semantics.
            pos_embed = self._encode_geometry(proposal).to(device=device, dtype=dtype)
            sem_embed = self._encode_semantic(proposal).to(device=device, dtype=dtype)

            symb = torch.cat((pos_embed, sem_embed), dim=1)
            x = self.symb_proj(symb)  # (N, hidden)

            # Lightweight context block.
            x = self._transformer_block(x)

            # Predict parameter offsets.
            offsets = self.param_offset_fc(x)  # (N,3)
            offsets_per_image.append(offsets)

        return offsets_per_image
