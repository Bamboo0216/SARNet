from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    encode_box_info,
    encode_orientedbox_info,
)


class RSCABlock(nn.Module):
    """
    Single geometry-aware attention block from the relation center to subject/object neighbors.

    Inputs:
      rel_feat  : [P, d_rel]
      subj_feat : [P, d_rel]
      obj_feat  : [P, d_rel]
      geom_rel  : [P, d_geom]
      geom_subj : [P, d_geom]
      geom_obj  : [P, d_geom]
    Output:
      rel_out   : [P, d_rel]
    """
    def __init__(self, d_rel: int = 1024, d_geom: int = 128, d_model: int = 512):
        super().__init__()
        self.d_rel = d_rel
        self.d_geom = d_geom
        self.d_model = d_model

        self.fc1 = nn.Linear(d_rel, d_model)
        self.fc2 = nn.Linear(d_model, d_rel)

        self.fc_delta = nn.Sequential(
            nn.Linear(d_geom, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        rel_feat,    # [P, d_rel]
        subj_feat,   # [P, d_rel]
        obj_feat,    # [P, d_rel]
        geom_rel,    # [P, d_geom]
        geom_subj,   # [P, d_geom]
        geom_obj,    # [P, d_geom]
    ):

        P = rel_feat.size(0)

        # neighbors: [P, 2, d_rel]
        neighbors = torch.stack([subj_feat, obj_feat], dim=1)
        pre = rel_feat  # residual

        # project to d_model
        x_center = self.fc1(rel_feat)  # [P, d_model]
        x_neighbors = self.fc1(neighbors.view(P * 2, self.d_rel)).view(P, 2, self.d_model)

        # q k v
        q = self.w_qs(x_center)        # [P, d_model]
        k = self.w_ks(x_neighbors)     # [P, 2, d_model]
        v = self.w_vs(x_neighbors)     # [P, 2, d_model]

        # geom pos encoding
        pos_enc_subj = self.fc_delta(geom_rel - geom_subj)  # [P, d_model]
        pos_enc_obj  = self.fc_delta(geom_rel - geom_obj)   # [P, d_model]
        pos_enc = torch.stack([pos_enc_subj, pos_enc_obj], dim=1)  # [P, 2, d_model]

        # attn logits
        attn = self.fc_gamma(q.unsqueeze(1) - k + pos_enc)          # [P, 2, d_model]
        attn = F.softmax(attn / math.sqrt(self.d_model), dim=1)     # softmax over 2 neighbors

        # aggregate
        msg = (attn * (v + pos_enc)).sum(dim=1)  # [P, d_model]
        rel_out = self.fc2(msg) + pre            # [P, d_rel]
        return rel_out


def _extract_geom_with_embeds(
    node_geom_embed: nn.Module,
    rel_geom_embed: nn.Module,
    proposals: List,
    union_proposals: List,
    rel_pair_idxs: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(proposals) == len(union_proposals) == len(rel_pair_idxs), \
        "proposals, union_proposals, and rel_pair_idxs must have the same batch length."

    geom_rel_list, geom_subj_list, geom_obj_list = [], [], []

    for b, (prop, uprop, pair_idx) in enumerate(zip(proposals, union_proposals, rel_pair_idxs)):
        assert pair_idx.dim() == 2 and pair_idx.size(1) == 2, f"rel_pair_idxs[{b}] must have shape [R_b, 2]."
        pair_idx = pair_idx.to(device).long()

        sub_idx = pair_idx[:, 0]
        obj_idx = pair_idx[:, 1]

        prop = prop.to(device)
        if prop.mode == "xywha":
            node_geom = node_geom_embed(encode_orientedbox_info([prop]))
        else:
            node_geom = node_geom_embed(encode_box_info([prop]))

        uprop = uprop.to(device)
        if uprop.mode == "xywha":
            rel_geom = rel_geom_embed(encode_orientedbox_info([uprop]))
        else:
            rel_geom = rel_geom_embed(encode_box_info([uprop]))

        assert rel_geom.size(0) == pair_idx.size(0), \
            f"union_proposals[{b}] must align one-to-one with rel_pair_idxs[{b}]."

        geom_rel_list.append(rel_geom)
        geom_subj_list.append(node_geom[sub_idx])
        geom_obj_list.append(node_geom[obj_idx])

    geom_rel = torch.cat(geom_rel_list, dim=0)
    geom_subj = torch.cat(geom_subj_list, dim=0)
    geom_obj = torch.cat(geom_obj_list, dim=0)
    return geom_rel, geom_subj, geom_obj


class RSCA(nn.Module):
    """
    Multi-layer wrapper following the Geometric_Group.forward aggregation pattern:

    Each layer receives the same relation feature input x0, then outputs are averaged.

    Inputs:
      subj_feat / obj_feat / rel_feat : [P,*] or [B,R,*]
      proposals / union_proposals     : List[BoxList], len=B
      rel_pair_idxs                  : List[Tensor], len=B, each [R_b, 2] (sub_idx,obj_idx) in proposals[b]

    Output:
      Same shape as rel_feat, either [P,d_rel] or [B,R,d_rel].
    """
    def __init__(
        self,
        num_layers: int = 4,
        d_node: int = 1024,
        d_rel: int = 1024,
        d_geom: int = 128,
        d_model: int = 512,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_node = d_node
        self.d_rel = d_rel
        self.d_geom = d_geom
        self.d_model = d_model

        # geom embedding
        self.node_geom_embed = nn.Sequential(
            nn.Linear(9, d_geom),
            nn.ReLU(),
            nn.Linear(d_geom, d_geom),
        )
        self.rel_geom_embed = nn.Sequential(
            nn.Linear(9, d_geom),
            nn.ReLU(),
            nn.Linear(d_geom, d_geom),
        )

        # node feat -> d_rel
        self.node_proj = nn.Linear(d_node, d_rel) if d_node != d_rel else None

        # blocks
        self.layers = nn.ModuleList([
            RSCABlock(d_rel=d_rel, d_geom=d_geom, d_model=d_model)
            for _ in range(num_layers)
        ])

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        subj_feat,             # [P,d_node] or [B,R,d_node]
        obj_feat,              # [P,d_node] or [B,R,d_node]
        rel_feat,              # [P,d_rel]  or [B,R,d_rel]
        proposals,             # len=B
        union_proposals,       # len=B
        rel_pair_idxs,         # len=B, each [R_b,2]
    ):

        subj_rep_list = []
        obj_rep_list = []

        for pair_idx, s_feat, o_feat in zip(rel_pair_idxs, subj_feat, obj_feat):
            subj_rep_list.append(s_feat[pair_idx[:, 0]])
            obj_rep_list.append(o_feat[pair_idx[:, 1]])

        subj_feat = torch.cat(subj_rep_list, dim=0)  # [sum_rel, C]
        obj_feat = torch.cat(obj_rep_list, dim=0)  # [sum_rel, C]

        rel_flat, subj_flat, obj_flat, orig_shape = self._flatten_feats(
            rel_feat=rel_feat,
            subj_feat=subj_feat,
            obj_feat=obj_feat,
            rel_pair_idxs=rel_pair_idxs,
        )

        # proj node feat if needed
        if self.node_proj is not None:
            subj_flat = self.node_proj(subj_flat)
            obj_flat  = self.node_proj(obj_flat)

        # geom (only once)
        geom_rel, geom_subj, geom_obj = self._extract_geom(
            proposals=proposals,
            union_proposals=union_proposals,
            rel_pair_idxs=rel_pair_idxs,
            device=rel_flat.device,
        )


        x0 = rel_flat
        feat_sum = None
        for i, layer in enumerate(self.layers):
            feat_i = layer(x0, subj_flat, obj_flat, geom_rel, geom_subj, geom_obj)
            if i == 0:
                feat_sum = feat_i
            else:
                feat_sum = feat_sum + feat_i

        out = feat_sum / float(self.num_layers)

        # reshape back
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], self.d_rel).contiguous()
        else:
            out = out.contiguous()
        return out

    def _flatten_feats(
        self,
        rel_feat: torch.Tensor,
        subj_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        rel_pair_idxs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
        """
        Flatten inputs to [P,*] and return orig_shape for reshaping the output.
        """
        orig_shape = rel_feat.shape

        B = len(rel_pair_idxs)
        pair_nums = [int(x.size(0)) for x in rel_pair_idxs]
        P = sum(pair_nums)

        if rel_feat.dim() == 3:
            assert rel_feat.size(0) == B, "rel_feat batch dimension must equal len(rel_pair_idxs)."
            R = rel_feat.size(1)
            assert all(r == R for r in pair_nums), \
                "When rel_feat has shape [B,R,*], every rel_pair_idxs[b] must have length R."

            rel_flat  = rel_feat.reshape(B * R, -1)
            subj_flat = subj_feat.reshape(B * R, -1)
            obj_flat  = obj_feat.reshape(B * R, -1)
        elif rel_feat.dim() == 2:
            assert rel_feat.size(0) == P, "The first dimension of rel_feat must equal sum_b R_b."
            rel_flat  = rel_feat
            subj_flat = subj_feat
            obj_flat  = obj_feat
        else:
            raise ValueError("rel_feat must be either 2D or 3D.")

        return rel_flat, subj_flat, obj_flat, orig_shape

    def _extract_geom(
        self,
        proposals: List,
        union_proposals: List,
        rel_pair_idxs: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract geometry features from proposals, union_proposals, and rel_pair_idxs:
          geom_rel, geom_subj, geom_obj
        All outputs are concatenated as [P, d_geom].

        Convention:
          union_proposals[b] order matches rel_pair_idxs[b] order.
        """
        return _extract_geom_with_embeds(
            self.node_geom_embed,
            self.rel_geom_embed,
            proposals,
            union_proposals,
            rel_pair_idxs,
            device,
        )
