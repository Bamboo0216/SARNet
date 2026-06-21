# -*- coding: utf-8 -*-
import torch
from typing import List, Tuple, Optional, Union, Any, Dict
from mmcv.ops import box_iou_rotated as _mmcv_iou

# ============================================================
# 旋转 IoU 选择：优先 mmcv.ops.box_iou_rotated，兜底为 AABB IoU
# ============================================================
def _get_rotated_iou_fn():
    info = {"name": "aabb"}
    try:
        def iou_mmcv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a: [Na,5], b: [Nb,5], (cx,cy,w,h,angle)，单位通常为度
            return _mmcv_iou(a, b)
        info["fn"] = iou_mmcv
        info["name"] = "mmcv"
        return info
    except Exception:
        pass

    # 回退：AABB IoU（将旋转框近似为水平框）
    def iou_aabb(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def to_aabb(x):
            cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            return torch.stack([x1, y1, x2, y2], dim=-1)
        a_aabb, b_aabb = to_aabb(a), to_aabb(b)
        Na, Nb = a_aabb.size(0), b_aabb.size(0)
        A = a_aabb[:, None, :].expand(Na, Nb, 4)
        B = b_aabb[None, :, :].expand(Na, Nb, 4)
        x1 = torch.maximum(A[..., 0], B[..., 0])
        y1 = torch.maximum(A[..., 1], B[..., 1])
        x2 = torch.minimum(A[..., 2], B[..., 2])
        y2 = torch.minimum(A[..., 3], B[..., 3])
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        areaA = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
        areaB = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
        return inter / (areaA + areaB - inter + 1e-9)

    info["fn"] = iou_aabb
    return info


# ============================================================
# rel_pair_idxs 归一化：返回 (N,2) long，缺失处为 -1
# 允许输入：
#   - Tensor(M,2)：按顺序对应 rboxes 的前 M 行（M 可≤N）
#   - (rows_idx, pairs) / {"rows":..., "pairs":...} 也支持（兼容）
# ============================================================
def _normalize_rel_pairs(
        N: int,
        rel_pair_idxs: Optional[Union[
            torch.Tensor,
            Dict[str, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor]
        ]],
        device: torch.device
) -> torch.Tensor:
    """
    返回 pairs_full: (N,2) long，缺失处为 -1。
    语义：第 i 行（联合框 i）对应的两两基础 box 索引（无先后）。
    """
    pairs_full = torch.full((N, 2), -1, dtype=torch.long, device=device)
    if rel_pair_idxs is None:
        return pairs_full

    def to_long_cuda(x):
        return x.to(device).long()

    # Tensor(M,2)
    if torch.is_tensor(rel_pair_idxs):
        if rel_pair_idxs.ndim != 2 or rel_pair_idxs.shape[1] != 2:
            raise ValueError("rel_pair_idxs 作为 Tensor 时必须是二维 (M,2)")
        M = rel_pair_idxs.shape[0]
        if M > 0:
            fill_M = min(M, N)
            pairs_full[:fill_M] = to_long_cuda(rel_pair_idxs[:fill_M])
            if M != N:
                print(f"[WARN] rel_pair_idxs 行数 M={M} 与 N={N} 不一致：仅前 {fill_M} 行被使用，其余为 -1。")
        return pairs_full

    # (rows,pairs) / {'rows':..., 'pairs':...}
    if isinstance(rel_pair_idxs, dict):
        rows = rel_pair_idxs.get("rows", None)
        pairs = rel_pair_idxs.get("pairs", None)
    elif isinstance(rel_pair_idxs, tuple) and len(rel_pair_idxs) == 2 and \
            torch.is_tensor(rel_pair_idxs[0]) and torch.is_tensor(rel_pair_idxs[1]):
        rows, pairs = rel_pair_idxs
    else:
        raise TypeError(
            "rel_pair_idxs 支持：Tensor(M,2)；或 (rows_idx, pairs)；或 {'rows': rows_idx, 'pairs': pairs}"
        )

    if not (torch.is_tensor(pairs) and pairs.ndim == 2 and pairs.shape[1] == 2):
        raise ValueError("rel_pair_idxs['pairs'] 必须是 (M,2)")

    if rows is None:
        M = pairs.shape[0]
        fill_M = min(M, N)
        pairs_full[:fill_M] = to_long_cuda(pairs[:fill_M])
        if M != N:
            print(f"[WARN] rel_pair_idxs['pairs'] 行数 M={M} 与 N={N} 不一致：仅前 {fill_M} 行被使用。")
        return pairs_full

    # 按 rows 离散写入
    if not (torch.is_tensor(rows) and rows.ndim == 1 and rows.shape[0] == pairs.shape[0]):
        raise ValueError("rows 必须是 (M,) 且与 pairs 行数一致")
    rows = to_long_cuda(rows)
    pairs = to_long_cuda(pairs)
    valid = (rows >= 0) & (rows < N)
    if valid.sum() < rows.numel():
        print("[WARN] rows 中存在越界索引，已自动过滤。")
    rows = rows[valid]
    pairs = pairs[valid]
    pairs_full[rows] = pairs
    return pairs_full


# ============================================================
# 辅助：适配 BoxList / Tensor
# ============================================================
def _extract_xywha_and_scores_from_boxlist(boxlist) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    兼容常见 BoxList/RotatedBoxList：
    - 几何：优先 .bbox 或 .tensor，要求 xywha
    - 分数：尝试 .get_field('scores'/'score'/'predict_logits') 或 .scores
    """
    # 几何
    if hasattr(boxlist, "bbox"):
        rboxes = boxlist.bbox
    elif hasattr(boxlist, "tensor"):
        rboxes = boxlist.tensor
    else:
        raise TypeError("Unsupported BoxList: no .bbox/.tensor")
    rboxes = rboxes.float().to(rboxes.device)

    # 分数（可选）
    scores = None
    try:
        if hasattr(boxlist, "get_field"):
            for k in ["scores", "score", "predict_logits"]:
                try:
                    s = boxlist.get_field(k)
                    if s is not None:
                        scores = s
                        break
                except Exception:
                    pass
        if scores is None and hasattr(boxlist, "scores"):
            scores = getattr(boxlist, "scores")
        if scores is not None and not torch.is_tensor(scores):
            scores = torch.as_tensor(scores)
        if scores is not None:
            scores = scores.float().to(rboxes.device)
    except Exception:
        scores = None
    return rboxes, scores


def _slice_boxlist(bl, keep_indices: torch.Tensor):
    """尽量返回同类型 BoxList 的子集；若失败，回退为几何张量."""
    try:
        return bl[keep_indices]
    except Exception:
        if hasattr(bl, "bbox"):
            return bl.bbox[keep_indices]
        if hasattr(bl, "tensor"):
            return bl.tensor[keep_indices]
        raise


def _boxlist_with_geometry(bl, geom: torch.Tensor):
    """
    构造一个“与 bl 同类的新实例”，几何替换为 geom，不修改 bl 本身。
    若失败则回退返回 geom 张量。
    """
    try:
        N = geom.size(0)
        idx = torch.arange(N, device=geom.device)
        new_bl = _slice_boxlist(bl, idx)   # 期望得到同类新对象
        # 覆盖几何字段
        if hasattr(new_bl, "bbox") and torch.is_tensor(getattr(new_bl, "bbox")):
            new_bl.bbox = geom
            return new_bl
        if hasattr(new_bl, "tensor") and torch.is_tensor(getattr(new_bl, "tensor")):
            new_bl.tensor = geom
            return new_bl
        # 如果没有上述字段，就直接回退
        return geom
    except Exception:
        return geom


# ============================================================
# 核心：面积优先的旋转框代理聚类（单 batch）
# 新增：proxy_boxes_full (N,5) —— 将代理几何回填到每个原始位置
#      base_sets_full (List[List[int]]) —— 每行对应的基础 box 集合
# ============================================================
@torch.no_grad()
def nms_rotated_with_proxy_area_first(
    rboxes: torch.Tensor,                 # (N,5) [cx,cy,w,h,angle]
    rel_pair_idxs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    scores: Optional[torch.Tensor] = None,# (N,) 可选，仅用于 gate（排序走面积）
    *,
    thresh: float = 0.5,                  # 基础 IoU 阈值
    score_gate: Optional[float] = None,   # 过滤极低分框（如 0.05），被过滤者保持自映射
    area_ratio_guard: float = 0.0,        # γ>0：仅当 area_i >= γ*area_j 时，i 可代理 j
    beta: float = 0.0,                    # >0：面积自适应阈值（大框阈值更低），建议 0~0.6
    thresh_low: float = 0.05,             # 自适应阈值下限
    # --- 可选加速开关 ---
    center_prefilter: bool = True,        # 是否启用中心/尺寸快速剔除
    center_margin: float = 1.0,           # 中心预过滤的松弛倍率（>=1）
    use_half: bool = True                 # mmcv路径下，IoU计算用半精度
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]], List[List[int]]
]:
    """
    返回：
      keep_indices:  (M,)   代理框的原始索引（按选择顺序）
      rep_boxes:     (M,5)  代理框几何
      proxy_of_full: (N,)   每个输入框被谁代理（自身为代理则等于自身索引；被 score_gate 过滤的为自映射）
      groups:        长度 M 的列表：各代理框覆盖到的“联合框原始索引列表”（含自身）
      rep_base_sets: 长度 M 的列表：各代理框覆盖到的“基础 box id 去重集合”（由 rel_pair_idxs 聚合）

    注：不再返回与输入数量一致的旋转框（proxy_boxes_full / base_sets_full 已移除）。
    """
    assert rboxes.dim() == 2 and rboxes.size(1) == 5, "rboxes 必须是 (N,5)"
    device = rboxes.device
    N = rboxes.size(0)

    # 关系对标准化
    pairs_full_all = _normalize_rel_pairs(N, rel_pair_idxs, device)  # (N,2) long；缺失为 -1

    if scores is None:
        scores = torch.zeros(N, device=device)

    # 分数门限（可选）
    valid = torch.ones(N, dtype=torch.bool, device=device)
    if score_gate is not None:
        valid &= (scores >= score_gate)

    rboxes_v = rboxes[valid]                  # (Nv,5)
    pairs_v  = pairs_full_all[valid]         # (Nv,2)
    idx_map  = torch.arange(N, device=device)[valid]  # (Nv,)
    Nv = rboxes_v.size(0)

    # 全被 gate：直接回传空的代表框与自映射
    if Nv == 0:
        keep_indices  = torch.empty(0, dtype=torch.long, device=device)
        rep_boxes     = rboxes.new_empty((0, 5))
        proxy_of_full = torch.arange(N, dtype=torch.long, device=device)  # 自映射
        groups: List[List[int]] = []
        rep_base_sets: List[List[int]] = []
        return keep_indices, rep_boxes, proxy_of_full, groups, rep_base_sets

    # 预取分量（减少索引开销）
    cx, cy, w, h, ang = (
        rboxes_v[:, 0], rboxes_v[:, 1],
        rboxes_v[:, 2], rboxes_v[:, 3],
        rboxes_v[:, 4]
    )

    areas = w * h
    order = torch.argsort(areas, descending=True)  # 面积优先

    iou_impl = _get_rotated_iou_fn()
    iou_fn = iou_impl["fn"]
    if iou_impl["name"] == "aabb":
        print("[WARN] 未检测到 mmcv.ops.box_iou_rotated，旋转 IoU 已降级为 AABB 近似。建议安装 mmcv。")

    suppressed = torch.zeros(Nv, dtype=torch.bool, device=device)
    proxy_of_local = torch.arange(Nv, dtype=torch.long, device=device)   # 在 valid 子集上的“被谁代理”
    keep_local: List[int] = []

    area_ref = areas.median()

    # 小工具：减少循环里的 .view 开销
    def _view1(x):  # x: (5,)
        return x.view(1, -1)

    # 预先计算半精度开关，避免循环里重复判断
    use_mmcv_half = (
        use_half
        and iou_impl["name"] == "mmcv"
        and rboxes_v.dtype == torch.float32
        and rboxes_v.is_cuda
    )

    for k, i in enumerate(order):
        if suppressed[i]:
            continue
        keep_local.append(int(i))

        rest = order[k + 1:]
        if rest.numel() == 0:
            continue

        # 面积自适应阈值（可选）
        if beta > 0:
            # (areas[i] / area_ref) ** (-beta)
            scale = (areas[i] / (area_ref + 1e-9)).pow(-beta)
            eff_thr_i = max(thresh_low, float(thresh * scale))
        else:
            eff_thr_i = thresh

        ii = rboxes_v[i]      # (5,)
        ii1 = _view1(ii)      # (1,5)

        # 候选：还未被抑制的
        cand = rest[~suppressed[rest]]
        if cand.numel() == 0:
            continue

        # ----------- 快速剔除阶段（可选）-----------
        if center_prefilter:
            # 以 AABB 的必要相交条件为快速过滤（对旋转框是宽松近似）
            # |dx| <= (wi + wj)/2 * margin，|dy| <= (hi + hj)/2 * margin
            dx = (cx[cand] - cx[i]).abs()
            dy = (cy[cand] - cy[i]).abs()
            half_w_sum = (w[cand] + w[i]) * 0.5 * center_margin
            half_h_sum = (h[cand] + h[i]) * 0.5 * center_margin
            pre_mask = (dx <= half_w_sum) & (dy <= half_h_sum)
            cand = cand[pre_mask]
            if cand.numel() == 0:
                continue
        # ------------------------------------------

        # 直接一次性计算当前框与所有候选框的 IoU
        if use_mmcv_half:
            ious = iou_fn(ii1.half(), rboxes_v[cand].half()).squeeze(0).float()
        else:
            ious = iou_fn(ii1, rboxes_v[cand]).squeeze(0)

        # 面积比保护
        cond = (ious >= eff_thr_i)
        if area_ratio_guard > 0:
            cond &= (areas[i] >= area_ratio_guard * areas[cand])

        hit = cand[cond]
        if hit.numel() > 0:
            suppressed[hit] = True
            proxy_of_local[hit] = i

    keep_local   = torch.tensor(keep_local, dtype=torch.long, device=device)
    keep_indices = idx_map[keep_local]      # 代表框的全局索引
    rep_boxes    = rboxes_v[keep_local]     # 代表框几何

    # 组成员与基础 box 聚合（全程在 GPU 上做，最后才转 list）
    groups: List[List[int]] = []
    rep_base_sets: List[List[int]] = []

    # keep_local 是 GPU 上的 long tensor，直接用它里的值做比较
    for r_loc in keep_local:
        # r_loc 是 0-dim tensor（在 device 上）
        members_loc = (proxy_of_local == r_loc).nonzero(as_tuple=True)[0]  # GPU 上完成
        if members_loc.numel() == 0:
            groups.append([])
            rep_base_sets.append([])
            continue

        # 映射回原始索引（转成 Python list）
        members_global = idx_map[members_loc].cpu().tolist()
        groups.append(members_global)

        # 聚合基础 box id，同样在 GPU 上算，再转 list
        pair_vals = pairs_v[members_loc].reshape(-1)
        pair_vals = pair_vals[pair_vals >= 0]
        if pair_vals.numel() == 0:
            rep_base_sets.append([])
        else:
            base_ids = torch.unique(pair_vals).cpu().tolist()
            rep_base_sets.append(base_ids)

    # 完整 proxy_of（含被 gate 的自映射）
    proxy_of_full = torch.arange(N, dtype=torch.long, device=device)
    proxy_of_full[valid] = idx_map[proxy_of_local]
    proxy_boxes_full = rboxes[proxy_of_full]

    # ========= 新增：把代表框的基础 box 集合，展开到所有 N 行 =========
    # 映射：代表框全局索引 -> 其基础 box 集合
    rep_global_idxs = keep_indices.tolist()                 # 长度 M
    rep_idx_to_base = {g_idx: rep_base_sets[t] for t, g_idx in enumerate(rep_global_idxs)}
    # base_sets_full[i]：第 i 行的代理代表框所覆盖到的所有基础 box（主+宾，去重）
    base_sets_full: List[List[int]] = []
    pof_cpu = proxy_of_full.tolist()
    for i in range(N):
        rep_g = pof_cpu[i]                                   # 第 i 行对应的代表框全局索引
        base_sets_full.append(rep_idx_to_base.get(rep_g, []))
    return keep_indices, rep_boxes, proxy_of_full, groups, rep_base_sets, proxy_boxes_full, base_sets_full



@torch.no_grad()
def proxy_cluster(
    proposals: Union[Any, torch.Tensor, List[Union[Any, torch.Tensor]]],
    rel_pair_idxs: Union[
        None,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        List[Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], None]]
    ],
    *,
    thresh: float = 0.5,
    score_gate: Optional[float] = None,
    area_ratio_guard: float = 0.0,
    beta: float = 0.0,
    thresh_low: float = 0.05,
):
    """
    统一入口：单/多 batch 均可调用（不做跨 batch 聚类）。

    说明：
      - 上游只关心代理框和代理关系，不再构造与输入数量一致的 proxy 框几何。
      - 返回结果中不再包含 `proxy_boxes_full` 和 `base_sets_full`。
    """

    assert isinstance(rel_pair_idxs, (list, tuple)) and len(proposals) == len(rel_pair_idxs), \
        "proposals 与 rel_pair_idxs 的 batch 数必须一致"

    results: List[Dict[str, Any]] = []

    for b in range(len(proposals)):
        # 取几何与分数
        if torch.is_tensor(proposals[b]):
            rboxes = proposals[b].float()
            scores = None
            bl = None
            rep_sliceable = False
        else:
            rboxes, scores = _extract_xywha_and_scores_from_boxlist(proposals[b])
            bl = proposals[b]
            rep_sliceable = True

        # 注意：这里调用的是已经精简过返回值、去掉 chunk_size 的版本
        keep_indices, rep_boxes, proxy_of, groups, rep_base_sets, proxy_boxes_full, base_sets_full = nms_rotated_with_proxy_area_first(
            rboxes=rboxes,
            rel_pair_idxs=rel_pair_idxs[b],
            scores=scores,
            thresh=thresh,
            score_gate=score_gate,
            area_ratio_guard=area_ratio_guard,
            beta=beta,
            thresh_low=thresh_low,
            # center_prefilter / center_margin / use_half 用默认值即可
        )

        # rep_out：尽量保持同类 BoxList
        rep_out = rep_boxes
        if rep_sliceable:
            try:
                rep_out = _slice_boxlist(bl, keep_indices)
            except Exception:
                # 万一 BoxList 切片失败，就退回到单纯的 rep_boxes 几何
                rep_out = rep_boxes

        proxy_boxes_full_out = _boxlist_with_geometry(bl, proxy_boxes_full)

        # 不再构造 proxy_boxes_full
        results.append({
            "keep_indices":   keep_indices,   # (M,)
            "rep":            rep_out,        # BoxList 或 tensor
            "rep_boxes":      rep_boxes,      # (M,5) tensor，永远有
            "proxy_of":       proxy_of,       # (N,)
            "groups":         groups,         # List[List[int]]
            "rep_base_sets":  rep_base_sets,  # List[List[int]]
            "proxy_boxes_full": proxy_boxes_full_out,
            "base_sets_full": base_sets_full
        })

    return results




from typing import List, Dict, Any, Union
import torch
import torch.nn as nn


class UnionSemanticMHA(nn.Module):
    """
    联合区域语义聚合（多查询多头注意力，兼容老版 nn.MultiheadAttention，无 batch_first）

    统一接口:
        forward(x, su_results, num_surround)

      参数:
        x: Tensor[N_all, d]
            - 单 batch: N_all = n
            - 多 batch: N_all = sum_b n_b（所有 batch 在 dim=0 上拼起来）

        su_results: list，长度为 B（batch 数）
            - 每个元素 su_results[b] 可以是:
                * dict，包含 "base_sets_full": List[List[int]]
                * 或直接是 List[List[int]]

        num_surround: List[int]
            - 长度必须为 B
            - 且 sum(num_surround) == N_all

      返回:
        out: Tensor[U_all, out_dim]
            - U_all = sum_b U_b
            - 多 batch 的输出按 batch 顺序在 dim=0 拼接
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        k_queries: int = 4,
        output_mode: str = "project",   # "concat" or "project"
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d = d_model
        self.k = k_queries
        self.output_mode = output_mode

        # K 个可学习查询
        self.queries = nn.Parameter(torch.randn(k_queries, d_model))

        # MultiheadAttention（输入形状 = [L, N, E]）
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, bias=True)

        if output_mode == "project":
            self.reduce = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),  # mean+max -> d
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )

    # -------- 单个联合区域：X=[T, d] -> [d] 或 [K*d] --------
    # 保留原始实现（方便单独调试 / 对比），在 batched 实现里不再逐个调用
    def forward_one(self, X: torch.Tensor) -> torch.Tensor:
        # 空区域：直接返回 0 向量
        if X.numel() == 0:
            out_dim = self.k * self.d if self.output_mode == "concat" else self.d
            return X.new_zeros(out_dim)

        # MultiheadAttention 需要 [L, N, E]，这里 batch=1
        Q = self.queries.unsqueeze(1)   # [K, 1, d]
        KV = X.unsqueeze(1)             # [T, 1, d]
        Y, _ = self.mha(Q, KV, KV, need_weights=False)  # [K, 1, d]
        Y = Y.squeeze(1)                # [K, d]

        if self.output_mode == "concat":
            return Y.flatten(0, 1)      # [K*d]
        else:
            y_mean = Y.mean(dim=0)                 # [d]
            y_max = Y.max(dim=0).values            # [d]
            return self.reduce(torch.cat([y_mean, y_max], dim=-1))  # [d]

    # -------- 从 base_sets_full 构造一个 batch 内的区域集合张量 --------
    def _regions_from_base_sets(
        self,
        edge_feats: torch.Tensor,
        base_sets_full: List[List[int]]
    ) -> List[torch.Tensor]:
        assert edge_feats.dim() == 2, "edge_feats 必须是 [n, d]"
        n, d = edge_feats.shape
        assert d == self.d, f"edge_feats 的最后一维 d={d} 与 d_model={self.d} 不一致"
        device = edge_feats.device

        regions: List[torch.Tensor] = []
        for ids in base_sets_full:
            idx = torch.as_tensor(ids, device=device, dtype=torch.long)
            if idx.numel() == 0:
                regions.append(edge_feats.new_zeros((0, d)))
                continue

            valid = (idx >= 0) & (idx < n)
            idx = idx[valid]
            if idx.numel() == 0:
                regions.append(edge_feats.new_zeros((0, d)))
                continue

            idx = torch.unique(idx)
            idx, _ = torch.sort(idx)    # 去重+排序
            regions.append(edge_feats.index_select(0, idx))   # [T_u, d]
        return regions

    # -------- 新增：对一个 batch 内所有联合区域一起做 MHA（加速关键） --------
    def forward_regions_batch(self, regions: List[torch.Tensor]) -> torch.Tensor:
        """
        regions: List[Tensor[T_u, d]]，长度 = U（本 batch 里 union 的个数）
        返回: Tensor[U, out_dim]
        语义等价于对每个 regions[u] 调一次 forward_one，但只做一次 MHA 调用。
        """
        device = self.queries.device
        d = self.d
        k = self.k
        out_dim = k * d if self.output_mode == "concat" else d

        U = len(regions)
        if U == 0:
            return self.queries.new_empty((0, out_dim))

        # 每个 union 的长度
        lens = torch.tensor([r.size(0) for r in regions],
                            device=device, dtype=torch.long)
        max_T = int(lens.max().item())

        # 全是空区域
        if max_T == 0:
            return self.queries.new_zeros((U, out_dim))

        # [T_max, U, d]，padding 成一个大 KV
        KV = self.queries.new_zeros((max_T, U, d))
        # key_padding_mask 形状 [U, T_max]，True = pad/ignore
        key_padding_mask = torch.ones(U, max_T, dtype=torch.bool, device=device)

        for u, r in enumerate(regions):
            T_u = r.size(0)
            if T_u == 0:
                continue
            KV[:T_u, u, :] = r
            key_padding_mask[u, :T_u] = False   # 有效位置

        # Q: [K, U, d]，把同一组 queries broadcast 给 U 个 union
        Q = self.queries.unsqueeze(1).expand(k, U, d)  # [K, 1, d] -> [K, U, d]

        # 一次性 MultiheadAttention
        Y, _ = self.mha(
            Q, KV, KV,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )  # [K, U, d]

        # 调整成 [U, K, d]
        Y = Y.permute(1, 0, 2).contiguous()  # [U, K, d]

        # 聚合 K 个查询
        if self.output_mode == "concat":
            out = Y.view(U, -1)   # [U, K*d]
        else:
            y_mean = Y.mean(dim=1)          # [U, d]
            y_max = Y.max(dim=1).values     # [U, d]
            out = self.reduce(torch.cat([y_mean, y_max], dim=-1))  # [U, d]

        # 对原本是空区域的 union，强制输出为 0，保持语义与 forward_one 一致
        empty_mask = (lens == 0)
        if empty_mask.any():
            out[empty_mask] = 0.0

        return out

    # -------- 统一 forward：单 / 多 batch 共用逻辑，输出拼在一起 --------
    def forward(
        self,
        x: torch.Tensor,
        su_results: List[Union[Dict[str, Any], List[List[int]]]],
        num_surround: List[int],
    ) -> torch.Tensor:
        """
        x: [N_all, d]
        su_results: list 长度 B
        num_surround: List[int]，len=B 且 sum(num_surround)=N_all
        """
        if not (isinstance(x, torch.Tensor) and x.dim() == 2):
            raise TypeError("x 必须是 2D Tensor [N_all, d]")

        N_all, d = x.shape
        if d != self.d:
            raise ValueError(f"x 的最后一维 d={d} 与 d_model={self.d} 不一致")

        if not isinstance(su_results, list):
            raise TypeError("su_results 必须是 list")
        B = len(su_results)

        if not isinstance(num_surround, (list, tuple)):
            raise TypeError("num_surround 必须是 List[int] 或类似序列类型")

        if len(num_surround) != B:
            raise ValueError(
                f"num_surround 长度 {len(num_surround)} 必须与 su_results 长度 B={B} 一致"
            )

        out_dim = self.k * self.d if self.output_mode == "concat" else self.d

        # 允许 B == 0：直接返回空
        if B == 0:
            return x.new_empty((0, out_dim))

        # 单 / 多 batch 共用循环，最后把各 batch 的结果 cat 在一起
        outs: List[torch.Tensor] = []
        start = 0
        for su_b, n_b in zip(su_results, num_surround):
            end = start + n_b
            x_b = x[start:end]  # [n_b, d]

            base_sets_b = su_b["base_sets_full"] if isinstance(su_b, dict) else su_b
            regions_b = self._regions_from_base_sets(x_b, base_sets_b)  # List[Tensor[T_u, d]]

            if len(regions_b) == 0:
                outs.append(x.new_empty((0, out_dim)))
            else:
                # ⚠️ 关键：对一个 batch 内所有 union 一次性做 MHA
                outs_b = self.forward_regions_batch(regions_b)  # [U_b, out_dim]
                outs.append(outs_b)

            start = end

        return torch.cat(outs, dim=0) if len(outs) > 0 else x.new_empty((0, out_dim))


@torch.no_grad()
def align_surround_features(
    su_results: List[Dict[str, Any]],
    surround_features: torch.Tensor,   # (sum_b M_b, C_s)
    union_features: torch.Tensor,      # (sum_b N_b, C_u) 仅用于长度/设备/类型对齐
) -> torch.Tensor:
    """
    根据 su_results 中的 proxy 信息，将按 batch 拼接的 surround_features
    展开对齐到所有原始框，返回 shape=(sum_b N_b, C_s) 的张量。

    参数：
      su_results: proxy_cluster 的输出 list，每个元素是一个 batch 的 dict，至少包含：
        - "keep_indices": (M_b,)  代理框的原始索引（相对该 batch）
        - "proxy_of":     (N_b,)  每个原始框被谁代理（相对该 batch）
      surround_features: 按 batch 顺序拼接的代理框特征，shape=(sum_b M_b, C_s)
      union_features:    按 batch 顺序拼接的原始框特征，shape=(sum_b N_b, C_u)

    返回：
      surround_features_full: shape=(sum_b N_b, C_s)，
        其中第 i 行对应 union_features[i] 的代理特征。
    """
    assert surround_features.dim() == 2 and union_features.dim() == 2
    device = union_features.device
    dtype = surround_features.dtype

    # 统计总 N、总 M 做一致性检查
    total_N_from_results = sum(int(res["proxy_of"].shape[0]) for res in su_results)
    total_M_from_results = sum(int(res["keep_indices"].shape[0]) for res in su_results)

    assert total_N_from_results == union_features.size(0), \
        f"union_features 行数 {union_features.size(0)} 与 su_results 中 proxy_of 总数 {total_N_from_results} 不一致"
    assert total_M_from_results == surround_features.size(0), \
        f"surround_features 行数 {surround_features.size(0)} 与 su_results 中 keep_indices 总数 {total_M_from_results} 不一致"

    N_total = union_features.size(0)
    C_s = surround_features.size(1)

    # 预先分配输出（与 union_features 对齐的顺序）
    surround_full = torch.empty((N_total, C_s), device=device, dtype=dtype)

    n_offset = 0  # 对应 union_features / proxy_of 的偏移
    m_offset = 0  # 对应 surround_features / keep_indices 的偏移

    for res in su_results:
        proxy_of_b: torch.Tensor = res["proxy_of"]      # (N_b,)
        keep_b: torch.Tensor = res["keep_indices"]      # (M_b,)

        N_b = int(proxy_of_b.shape[0])
        M_b = int(keep_b.shape[0])

        # 当前 batch 对应的特征切片
        surround_b = surround_features[m_offset:m_offset + M_b]  # (M_b, C_s)

        # 保证都在同一设备
        proxy_of_b = proxy_of_b.to(device)
        keep_b = keep_b.to(device)

        # 构建：原始索引 -> 代理框在本 batch 的局部位置 [0..M_b-1]
        # 初始化为 -1（理论上 proxy_of 都应该落在 keep_b 里，保险起见先设为 -1）
        rep_pos = proxy_of_b.new_full((N_b,), -1, dtype=torch.long)   # (N_b,)
        rep_pos[keep_b] = torch.arange(M_b, device=device, dtype=torch.long)

        # 每个原始框对应的代理框局部索引
        local_rep_idx = rep_pos[proxy_of_b]   # (N_b,)

        # 如果存在 rep_pos 为 -1 的情况（例如 score_gate 剪掉导致某些框从未当过代理），
        # 我们将这些位置的 surround 置零。
        invalid_mask = (local_rep_idx < 0)
        # 避免索引负数，先 clamp 再赋零
        local_rep_idx_clamped = local_rep_idx.clamp(min=0)

        surround_full_b = surround_b[local_rep_idx_clamped]  # (N_b, C_s)
        if invalid_mask.any():
            surround_full_b[invalid_mask] = 0

        # 写回总输出
        surround_full[n_offset:n_offset + N_b] = surround_full_b

        n_offset += N_b
        m_offset += M_b

    return surround_full