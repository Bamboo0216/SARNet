from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import encode_orientedbox_info
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union
import torch
def get_union_proposals(proposals, rel_pair_idxs):
    # union_proposals = []
    # start_time = time.time()
    for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
        head_proposal = proposal[rel_pair_idx[:, 0]]
        tail_proposal = proposal[rel_pair_idx[:, 1]]

    # union_proposal = boxlist_union(head_proposal, tail_proposal, flag2=True)
    # union_proposals.append(union_proposal)
    union_info = torch.cat((encode_orientedbox_info([head_proposal]), encode_orientedbox_info([tail_proposal])), dim=1)
    return union_info

def encode_union_proposals(proposals, rel_pair_idxs):
    union_info = get_union_proposals(proposals, rel_pair_idxs)
    # return encode_orientedbox_info(union_proposals)
    return union_info
