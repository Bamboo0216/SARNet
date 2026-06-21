import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


class Get_Atten_map_mc_clear(nn.Module):

    def __init__(self, input_dims, p):
        super(Get_Atten_map_mc_clear, self).__init__()
        self.input_dims = input_dims
        self.p = p
        self.ws = nn.Linear(self.input_dims, self.input_dims)
        self.wo = nn.Linear(self.input_dims, self.input_dims)
        self.w = nn.Linear(self.input_dims, self.p)
        # self.act = nn.ReLU(inplace=True)
        # self.act = CELU(alpha=1.3)
        # self.act = nn.Sequential()
        self.tau = 0.5
        self.tau_pm2 = 4.
        self.T = 1.

    def forward(self, obj_feats, union_feats, pair_idxs):
        norm_mat = (obj_feats[:, None, :] - obj_feats[None, :, :]).norm(dim=-1, keepdim=True)
        n_nodes = obj_feats.shape[0]
        atten_f = self.w(self.ws(obj_feats)[pair_idxs[:, 0]] * self.wo(obj_feats)[pair_idxs[:, 1]] * union_feats)
        atten_tensor = torch.zeros(n_nodes, n_nodes, self.p).to(obj_feats)
        atten_tensor[pair_idxs[:, 0], pair_idxs[:, 1]] += atten_f
        eye_tensor = -torch.eye(n_nodes).unsqueeze(-1).repeat(1, 1, self.p).to(obj_feats) * 1e4
        atten_tensor = atten_tensor + eye_tensor
        atten_map = F.softmax(atten_tensor, dim=1)

        Omega = torch.zeros_like(atten_map)
        Omega = Omega.masked_fill_(norm_mat < self.tau, self.tau_pm2)
        Omega = Omega.masked_fill_(torch.eye(Omega.shape[0], dtype=bool, device=Omega.device).unsqueeze(-1), 0.)
        Omega = torch.where((norm_mat >= self.tau) & (norm_mat < self.T), norm_mat.clamp(min=1e-5).pow(-2.), Omega)
        atten_map = Omega * atten_map

        return atten_map

class Get_Atten_map_mc(nn.Module):

    def __init__(self, input_dims, p):
        super(Get_Atten_map_mc, self).__init__()
        self.input_dims = input_dims
        self.p = p
        self.ws = nn.Linear(input_dims, input_dims)
        self.wo = nn.Linear(input_dims, input_dims)
        self.w = nn.Linear(self.input_dims, self.p)

    def forward(self, obj_feats, rel_inds, union_feats, n_nodes):
        atten_f = self.w(self.ws(obj_feats)[rel_inds[:, 1]] * \
                         self.wo(obj_feats)[rel_inds[:, 2]] * union_feats)
        atten_tensor = Variable(torch.zeros(n_nodes, n_nodes, self.p)).cuda().float()
        head = rel_inds[:, 1:].min()
        atten_tensor[rel_inds[:, 1] - head, rel_inds[:, 2] - head] += atten_f
        atten_tensor = F.sigmoid(atten_tensor)
        atten_tensor = atten_tensor * (
                    1 - Variable(torch.eye(n_nodes).float()).unsqueeze(-1).repeat(1, 1, self.p).cuda())
        return atten_tensor / torch.sum(atten_tensor, 1)


def mc_matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)


class LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-5):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.

        Thanks to CyberZHG's code in https://github.com/CyberZHG/torch-layer-normalization.git .
        """
        super(LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )


class Direction_Aware_MP(nn.Module):

    def __init__(self, input_dims):
        super(Direction_Aware_MP, self).__init__()
        self.input_dims = input_dims
        self.trans = nn.Sequential(nn.Linear(self.input_dims, input_dims // 4),
                                   LayerNorm(self.input_dims // 4), nn.ReLU(inplace=True),
                                   nn.Linear(self.input_dims // 4, self.input_dims))

        self.get_atten_tensor = Get_Atten_map_mc(self.input_dims, p=1)

        self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims // 2),
                                  nn.ReLU(inplace=True))
        # self.conv = nn.Linear(self.input_dims, self.input_dims // 4) # use rel in the end.

    def forward(self, obj_feats, phr_feats, im_inds, rel_inds):
        num_img = int(im_inds.max()) + 1
        obj_indices_sets = [torch.nonzero(im_inds == i).data.squeeze() for i in range(num_img)]
        obj2obj_feats_sets = []
        rel_indices_sets = [torch.nonzero(rel_inds[:, 0] == i).squeeze() for i in range(num_img)]

        for i, obj_indices in enumerate(obj_indices_sets):
            entities_num = obj_indices.size(0)
            cur_obj_feats = obj_feats[obj_indices]
            rel_indices = rel_indices_sets[i]
            atten_tensor = self.get_atten_tensor(obj_feats, rel_inds[rel_indices], phr_feats[rel_indices], entities_num)
            atten_tensor_t = torch.transpose(atten_tensor, 1, 0)
            atten_tensor = torch.cat((atten_tensor, atten_tensor_t), -1)
            context_feats = mc_matmul(atten_tensor, self.conv(cur_obj_feats))
            obj2obj_feats_sets.append(self.trans(context_feats))

        return F.relu(obj_feats + torch.cat(obj2obj_feats_sets, 0), inplace=True)

class Message_Passing4OBJ(nn.Module):

    def __init__(self, input_dims):
        super(Message_Passing4OBJ, self).__init__()
        self.input_dims = input_dims
        self.trans = nn.Sequential(nn.Linear(self.input_dims, self.input_dims*2),
                                   nn.ReLU(),
                                   nn.Linear(self.input_dims*2, self.input_dims))

        self.get_atten_tensor = Get_Atten_map_mc_clear(self.input_dims, p=1)

        self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims),
                                    nn.ReLU())

        self.ln1 = nn.LayerNorm(self.input_dims)
        self.ln2 = nn.LayerNorm(self.input_dims)

    def forward(self, obj_feats, phr_feats, pair_idxs):

        refined_obj_feats = []

        for iobj_feats, iphr_feats, ipair_idxs in zip(
            obj_feats, phr_feats, pair_idxs
        ):
            if not ipair_idxs.shape[0] > 1:
                refined_obj_feats.append(
                    iobj_feats
                )
                continue
            atten_tensor = self.get_atten_tensor(iobj_feats, iphr_feats, ipair_idxs)

            context_feats = torch.mm(atten_tensor.squeeze(-1), self.conv(self.ln1(iobj_feats)))

            outputs = iobj_feats + context_feats

            refined_obj_feats.append(
                F.relu(
                    outputs + self.trans(self.ln2(outputs))
                )
            )
        return refined_obj_feats