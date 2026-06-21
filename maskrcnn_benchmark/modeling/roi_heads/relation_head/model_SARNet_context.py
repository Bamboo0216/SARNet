'''Rectified Identity Cell'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info
from torch.cuda import device

from .utils_motifs import encode_orientedbox_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import nms_overlaps_rotated

class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats=None, num_objs=None):
        assert num_objs is not None
        outp = self.SA_transformer_encoder(x, num_objs)

        return outp

class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Cross_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats, num_objs=None):
        assert num_objs is not None
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)

        return outp

class Single_Layer_Hybrid_Attention(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        textual_output = tsa + tca
        visual_output = vsa + vca

        return visual_output, textual_output

class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats
        textual_output = text_feats

        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

        visual_output = visual_output + textual_output

        return visual_output, textual_output

class SARNetContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        self.type = self.cfg.Type
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.context_obj = SHA_Encoder(config, self.obj_layer)
        self.context_edge = RelationEncoder(config, self.edge_layer)

    def forward(self, roi_features, proposals, union_features, union_proposals, rel_pair_idxs,logger=None):
        union_output = union_features
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()  # Detector logits from proposals.
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        if proposals[0].bbox.shape[-1] == 5: 
            assert proposals[0].mode == 'xywha'
            TEMP = encode_orientedbox_info(proposals)
           
           
        else:
            assert proposals[0].mode == 'xyxy'
            TEMP = encode_box_info(proposals)
          

        # bbox embedding will be used as input
        pos_embed = self.bbox_embed(TEMP)

        # encode objects with transformer

        num_objs = [len(p) for p in proposals]
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)
        obj_feats_vis, _, = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)#   object encoder
        obj_feats = obj_feats_vis

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_labels)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_preds)

        # Update proposal predictions required by AEIP message passing.
        lengths = [len(p) for p in proposals]
        probs = F.softmax(obj_dists, dim=-1)
        idx = torch.arange(obj_dists.size(0), device=obj_dists.device)
        # Keep the score for each box's predicted label.
        pred_scores = probs[idx, obj_preds]

        score_splits = pred_scores.split(lengths, dim=0)
        logit_splits = obj_dists.split(lengths, dim=0)
        label_splits = obj_preds.split(lengths, dim=0)

        for i, (scores, logits, labels) in enumerate(
                zip(score_splits, logit_splits, label_splits)):
            proposals[i].add_field('pred_scores', scores)
            proposals[i].add_field('predict_logits', logits)
            proposals[i].add_field('pred_labels', labels)

        # edge context
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis)#  object decoder
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt)
        edge_ctx_vis, edge_ctx_txt, union_ctx_vis = self.context_edge(
            edge_pre_rep_vis,
            edge_pre_rep_txt,
            num_objs,
            union_features,
            proposals,
            union_proposals,
            rel_pair_idxs,
        )  # relation encoder
        union_output = union_ctx_vis

        edge_ctx = edge_ctx_vis
        union_ctx = union_output

        return obj_dists, obj_preds, edge_ctx, union_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            if ("HBB" in self.type) or ("CV" in self.type) :
               is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)
            else:
               is_overlap = nms_overlaps_rotated(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

class RelationEncoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])

        from maskrcnn_benchmark.modeling.roi_heads.relation_head.rel_proposal_network.model_GaussGate import GaussGate
        self.shared_gauss_gate = GaussGate(
            lam_mode="normalized",
        )  # shared GaussGate relation prior

        from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_AEIP import AEIP
        self.aeip_modules = nn.ModuleList([
            AEIP(config, self.hidden_dim, use_shared_gates=True)
            for _ in range(n_layers)
        ])

    def forward(self, visual_feats, text_feats, num_objs, union_features, proposals, union_proposals, rel_pair_idxs):
        visual_output = visual_feats
        textual_output = text_feats
        union_output = union_features

        shared_gauss_scores = self.shared_gauss_gate(proposals, rel_pair_idxs)
        for enc_layer, aeip in zip(self.cross_module, self.aeip_modules):
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

            visual_output,  union_output = aeip(
                visual_output,
                union_output,
                proposals,
                union_proposals,
                rel_pair_idxs,
                shared_gauss_scores = shared_gauss_scores,
            )

        visual_output = visual_output + textual_output
        union_features = union_output

        return visual_output, textual_output, union_features

def build_pairwise_ctx(edge_ctx_txt, rel_pair_idxs, num_objs):
    out_chunks = []
    base = 0  # Batch offset in edge_ctx_txt.

    for pairs, n_obj in zip(rel_pair_idxs, num_objs):
        if pairs.numel() == 0:
            base += n_obj
            continue

        pairs = pairs.long().to(edge_ctx_txt.device)
        idx_s = base + pairs[:, 0]
        idx_o = base + pairs[:, 1]

        subj = edge_ctx_txt.index_select(0, idx_s)  # (Ri, C)
        obj  = edge_ctx_txt.index_select(0, idx_o)  # (Ri, C)
        out_chunks.append(torch.cat([subj, obj], dim=1))  # (Ri, 2C)

        base += n_obj

    if len(out_chunks) == 0:
        return edge_ctx_txt.new_zeros((0, edge_ctx_txt.size(1) * 2))

    return torch.cat(out_chunks, dim=0)

if __name__ == '__main__':
    pass
