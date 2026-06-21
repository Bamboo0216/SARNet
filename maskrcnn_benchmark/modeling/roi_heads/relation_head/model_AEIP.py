import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor
from maskrcnn_benchmark.modeling.roi_heads.relation_head.rel_proposal_network.model_GaussGate import GaussGate


class GatedMessagePassingUnit(nn.Module):
    def __init__(self, input_dim, filter_dim=64):
        """
        Args:
            input_dim: Feature dimension of each node.
            filter_dim: Output width of the edge MLP before reducing to a scalar gate.
        """
        super(GatedMessagePassingUnit, self).__init__()
        self.w = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, filter_dim, bias=True),
        )

        self.fea_size = input_dim
        self.filter_size = filter_dim

        self.gate_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
        self.aux_gate_weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, unary_term, pair_term, aux_gate=None):
        """
        Generate a gated message from source features to target features.
        Args:
            unary_term: Target node features with shape (E, D), (1, D), or (E, D).
            pair_term: Source node features with shape (E, D), (1, D), or (E, D).
            aux_gate: Auxiliary gate from the GaussGate prior with shape (E,), (E, 1), or (E, D').

        Returns:
            output: (E, D)  = pair_term * gate
            gate:   (E,)    Fused scalar gate.
        """
        if unary_term.size(0) == 1 and pair_term.size(0) > 1:
            unary_term = unary_term.expand(pair_term.size(0), unary_term.size(1))
        if unary_term.size(0) > 1 and pair_term.size(0) == 1:
            pair_term = pair_term.expand(unary_term.size(0), pair_term.size(1))

        paired_feats = torch.cat([unary_term, pair_term], dim=1)

        gate = torch.sigmoid(self.w(paired_feats))
        if gate.dim() > 1:
            gate = gate.mean(dim=1)

        if aux_gate is not None:
            if aux_gate.dim() > 1:
                aux_gate = aux_gate.mean(dim=1) if aux_gate.size(1) > 1 else aux_gate.view(-1)

            g_net_logit = torch.logit(gate, eps=1e-6)
            g_aux_logit = torch.logit(aux_gate, eps=1e-6)

            fused_logit = self.gate_weight * g_net_logit + self.aux_gate_weight * g_aux_logit
            gate = torch.sigmoid(fused_logit)        # (E,)

        output = pair_term * gate.view(-1, 1)

        return output, gate


class MessageFusion(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MessageFusion, self).__init__()
        self.wih = nn.Linear(input_dim, input_dim, bias=True)
        self.whh = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output


class AEIP(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        hidden_dim=1024,
        num_iter=2,
        dropout=False,
        gate_width=128,
        use_shared_gates=False,
    ):
        super(AEIP, self).__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.update_step = num_iter
        self.use_shared_gates = use_shared_gates

        if self.update_step < 1:
            print(
                "WARNING: the update_step should be greater than 0, current: ",
                +self.update_step,
            )
        self.pooling_dim = 4096

        self.use_valid_pair_filtering = True
        self.pretrain_relation_classifier_mode = False

        self.message_passing_refine_iters = 1

        self.rel_input_proj = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim),
            nn.ReLU(True),
        )

        self.share_parameters_each_iter = (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SHARE_PARAMETERS_EACH_ITER
        )

        num_parameter_sets = num_iter
        if self.share_parameters_each_iter:
            num_parameter_sets = 1
        self.subject_to_predicate_gate = nn.Sequential(
            *[GatedMessagePassingUnit(self.hidden_dim, gate_width) for _ in range(num_parameter_sets)]
        )
        self.object_to_predicate_gate = nn.Sequential(
            *[GatedMessagePassingUnit(self.hidden_dim, gate_width) for _ in range(num_parameter_sets)]
        )
        self.predicate_to_subject_gate = nn.Sequential(
            *[GatedMessagePassingUnit(self.hidden_dim, gate_width) for _ in range(num_parameter_sets)]
        )
        self.predicate_to_object_gate = nn.Sequential(
            *[GatedMessagePassingUnit(self.hidden_dim, gate_width) for _ in range(num_parameter_sets)]
        )

        self.object_message_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(num_parameter_sets)]
        )  #
        self.predicate_message_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(num_parameter_sets)]
        )

        self.obj_input_proj = nn.Sequential(
            make_fc(512, self.hidden_dim),
            nn.ReLU(True),
        )

        self.obj_output_proj = nn.Sequential(
            make_fc(self.hidden_dim, 512),
            nn.ReLU(True),
        )

        self.rel_output_proj = nn.Sequential(
            make_fc(self.hidden_dim, 4096),
            nn.ReLU(True),
        )

        if not self.use_shared_gates:
            self.local_gauss_gate = GaussGate(lam_mode="normalized")

    def set_pretrain_relation_classifier_mode(self, val=True):
        self.pretrain_relation_classifier_mode = val

    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs, relatedness):
        """
        prepare the index of how subject and object related to the union boxes
        :param num_proposals:
        :param rel_pair_idxs:
        :return:
            ALL RETURN THINGS ARE BATCH-WISE CONCATENATED

            rel_inds,
                extent the instances pairing matrix to the batch wised (num_rel, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
            selected_relness,
                relatedness scores for selected relation proposals (val_rel_pair_num, 1)
            selected_rel_pair_indices,
                selected relation proposal indices used for message passing
        """
        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]
        pair_relness_batch = []

        for idx, (prop, rel_ind_i) in enumerate(
            zip(
                proposals,
                rel_pair_idxs,
            )
        ):
            assert relatedness is not None
            related_matrix = relatedness[idx]
            pair_relness = related_matrix[rel_ind_i[:, 0], rel_ind_i[:, 1]]
            pair_relness_batch.append(pair_relness)
            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)

        subj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        obj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        if len(pair_relness_batch) != 0:
            offset = 0
            selected_pair_indices = []
            for each_img_relness in pair_relness_batch:
                selected_rel_pair_indices = squeeze_tensor(
                    torch.nonzero(each_img_relness > 0.0001)
                )
                selected_pair_indices.append(
                    selected_rel_pair_indices + offset
                )
                offset += len(each_img_relness)

            selected_rel_pair_indices = torch.cat(selected_pair_indices, 0)
            pair_relness_cat = torch.cat(
                pair_relness_batch, 0
            )

            subj_pred_map[
                rel_inds_batch_cat[selected_rel_pair_indices, 0],
                selected_rel_pair_indices,
            ] = 1
            obj_pred_map[
                rel_inds_batch_cat[selected_rel_pair_indices, 1],
                selected_rel_pair_indices,
            ] = 1
            selected_relness = pair_relness_cat
        else:
            # or all relationship pairs
            selected_rel_pair_indices = torch.arange(
                len(rel_inds_batch_cat[:, 0]), device=rel_inds_batch_cat.device
            )
            selected_relness = None
            subj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 0].contiguous().view(1, -1)), 1)
            obj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 1].contiguous().view(1, -1)), 1)
        return (
            rel_inds_batch_cat,
            subj_pred_map,
            obj_pred_map,
            selected_relness,
            selected_rel_pair_indices,
        )

    def prepare_message(
            self,
            target_features,
            source_features,
            select_mat,
            gate_module,
            relness_scores=None,
    ):
        """
        generate the message from the source nodes for the following merge operations.

        :param target_features: (num_inst, dim)
        :param source_features: (num_rel, dim)
        :param select_mat:  (num_inst, rel_pair_num)
        :param gate_module:
        :param relness_scores: (num_rel, ) GaussGate relation scores in [0, 1].

        :return: (num_inst, dim)
        """

        if select_mat.sum() == 0:
            return torch.zeros_like(target_features)

        transfer_list = (select_mat > 0).nonzero()
        source_indices = transfer_list[:, 1]
        target_indices = transfer_list[:, 0]

        source_f = torch.index_select(source_features, 0, source_indices)
        target_f = torch.index_select(target_features, 0, target_indices)

        assert relness_scores is not None, "AEIP requires GaussGate relness scores."
        select_relness = relness_scores.index_select(0, source_indices)  # (E,)
        transferred_features, weighting_gate = gate_module(
            target_f, source_f, select_relness
        )

        if weighting_gate.dim() > 1:
            weighting_gate = weighting_gate.view(-1)

        eps = 1e-6
        edge_weights = (weighting_gate * select_relness).clamp(min=eps, max=1.0)  # (E,)

        aggregator_matrix = torch.zeros(
            (target_features.shape[0], transferred_features.shape[0]),
            dtype=transferred_features.dtype,
            device=transferred_features.device,
        )

        for f_id in range(target_features.shape[0]):
            if select_mat[f_id, :].sum() > 0:
                feature_indices = squeeze_tensor((transfer_list[:, 0] == f_id).nonzero())
                aggregator_matrix[f_id, feature_indices] = edge_weights.index_select(0, feature_indices)

        aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)

        norm = aggregator_matrix.sum(dim=1)  # (target,)
        valid = norm != 0
        norm = norm.unsqueeze(1).expand(norm.shape[0], aggregate_feat.shape[1])
        aggregate_feat[valid] /= norm[valid]

        return aggregate_feat

    def forward(
        self,
        inst_features,
        rel_union_features,
        proposals,
        union_proposals,
        rel_pair_inds,
        rel_gt_binarys=None,
        logger=None,
        shared_gauss_gate=None,
        shared_gauss_scores=None,
    ):
        """

        :param inst_features: instance_num, pooling_dim
        :param rel_union_features:  rel_num, pooling_dim
        :param proposals: instance proposals
        :param rel_pair_inds: relation pair indices list(tensor)
        :return:
        """

        obj_features = inst_features
        rel_features = rel_union_features

        refined_rel_features_by_iter = [rel_features]
        refined_obj_features_by_iter = [obj_features]

        for refine_iter in range(self.message_passing_refine_iters):
            if shared_gauss_scores is not None:
                relatedness_scores = shared_gauss_scores
            else:
                gauss_gate = shared_gauss_gate if shared_gauss_gate is not None else getattr(self, "local_gauss_gate", None)
                assert gauss_gate is not None, "AEIP requires GaussGate (provide shared_gauss_gate or keep self.local_gauss_gate)."
                relatedness_scores = gauss_gate(proposals, rel_pair_inds)

            obj_features_by_iter = [
                self.obj_input_proj(obj_features),
            ]
            rel_features_by_iter = [
                self.rel_input_proj(rel_features),
            ]

            valid_inst_idx = []
            if self.use_valid_pair_filtering:
                for p in proposals:
                    valid_inst_idx.append(p.get_field("pred_scores") > 0.03)

            if len(valid_inst_idx) > 0:
                valid_inst_idx = torch.cat(valid_inst_idx, 0)
            else:
                valid_inst_idx = torch.zeros(0)

            if self.pretrain_relation_classifier_mode:
                refined_inst_features = obj_features_by_iter[-1]
                refined_rel_features = rel_features_by_iter[-1]

                refined_obj_features_by_iter.append(refined_inst_features)
                refined_rel_features_by_iter.append(refined_rel_features)
                continue

            else:

                (
                    batchwise_rel_pair_inds,
                    subj_pred_map,
                    obj_pred_map,
                    relness_scores,
                    _selected_rel_pair_indices,
                ) = self._prepare_adjacency_matrix(
                    proposals, rel_pair_inds, relatedness_scores,
                )

                if (
                    len(squeeze_tensor(valid_inst_idx.nonzero())) < 1
                    or len(squeeze_tensor(batchwise_rel_pair_inds.nonzero())) < 1
                    or len(squeeze_tensor(subj_pred_map.nonzero())) < 1
                    or len(squeeze_tensor(obj_pred_map.nonzero())) < 1
                    or self.pretrain_relation_classifier_mode
                ):
                    refined_inst_features = obj_features_by_iter[-1]
                    refined_rel_features = rel_features_by_iter[-1]

                    refined_obj_features_by_iter.append(refined_inst_features)
                    refined_rel_features_by_iter.append(refined_rel_features)

                    continue

            for t in range(self.update_step):
                parameter_idx = 0
                if not self.share_parameters_each_iter:
                    parameter_idx = t
                object_sub = self.prepare_message(
                    obj_features_by_iter[t],
                    rel_features_by_iter[t],
                    subj_pred_map,
                    self.predicate_to_subject_gate[parameter_idx],
                    relness_scores=relness_scores,
                )
                object_obj = self.prepare_message(
                    obj_features_by_iter[t],
                    rel_features_by_iter[t],
                    obj_pred_map,
                    self.predicate_to_object_gate[parameter_idx],
                    relness_scores=relness_scores,
                )

                object_message = (object_sub + object_obj) / 2.0
                obj_features_by_iter.append(
                    obj_features_by_iter[t]
                    + self.object_message_fusion[parameter_idx](
                        object_message, obj_features_by_iter[t]
                    )
                )

                indices_sub = batchwise_rel_pair_inds[:, 0]
                indices_obj = batchwise_rel_pair_inds[:, 1]  # num_rel, 1

                if self.use_valid_pair_filtering:
                    valid_sub_inst_in_pairs = valid_inst_idx[indices_sub]
                    valid_obj_inst_in_pairs = valid_inst_idx[indices_obj]
                    valid_inst_pair_inds = (valid_sub_inst_in_pairs) & (
                        valid_obj_inst_in_pairs
                    )
                    indices_sub = indices_sub[valid_inst_pair_inds]
                    indices_obj = indices_obj[valid_inst_pair_inds]

                    feat_sub2pred = torch.index_select(obj_features_by_iter[t], 0, indices_sub)
                    feat_obj2pred = torch.index_select(obj_features_by_iter[t], 0, indices_obj)
                    valid_pairs_rel_feats = torch.index_select(
                        rel_features_by_iter[t],
                        0,
                        squeeze_tensor(valid_inst_pair_inds.nonzero()),
                    )

                    phrase_sub, _ = self.subject_to_predicate_gate[parameter_idx](
                        valid_pairs_rel_feats, feat_sub2pred
                    )
                    phrase_obj, _ = self.object_to_predicate_gate[parameter_idx](
                        valid_pairs_rel_feats, feat_obj2pred
                    )
                    predicate_message = (phrase_sub + phrase_obj) / 2.0
                    next_rel_features = self.predicate_message_fusion[parameter_idx](
                        predicate_message, valid_pairs_rel_feats
                    )

                    updated_rel_features = rel_features_by_iter[t].clone()
                    updated_rel_features[
                        valid_inst_pair_inds
                    ] += next_rel_features

                    rel_features_by_iter.append(updated_rel_features)
                else:

                    feat_sub2pred = torch.index_select(obj_features_by_iter[t], 0, indices_sub)
                    feat_obj2pred = torch.index_select(obj_features_by_iter[t], 0, indices_obj)
                    phrase_sub, _ = self.subject_to_predicate_gate[parameter_idx](
                        rel_features_by_iter[t], feat_sub2pred
                    )
                    phrase_obj, _ = self.object_to_predicate_gate[parameter_idx](
                        rel_features_by_iter[t], feat_obj2pred
                    )
                    predicate_message = (phrase_sub + phrase_obj) / 2.0
                    rel_features_by_iter.append(
                        rel_features_by_iter[t]
                        + self.predicate_message_fusion[parameter_idx](
                            predicate_message, rel_features_by_iter[t]
                        )
                    )
            refined_inst_features = obj_features_by_iter[-1]
            refined_rel_features = rel_features_by_iter[-1]

            refined_obj_features_by_iter.append(refined_inst_features)
            refined_rel_features_by_iter.append(refined_rel_features)

        return self.obj_output_proj(refined_obj_features_by_iter[-1]), self.rel_output_proj(refined_rel_features_by_iter[-1])


def build_aeip_model(cfg, in_channels):
    return AEIP(cfg, in_channels)
