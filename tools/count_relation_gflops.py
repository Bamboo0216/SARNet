# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Count approximate GFLOPs for predcls scene graph relation inference.

This script follows the same config precedence as the training scripts:

    defaults.py -> --config-file -> opts

Only predcls is supported in this first version:

    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True

The counter uses lightweight forward hooks and selected torch functional
wrappers. It reports counted theoretical FLOPs, with 1 MAC = 1 FLOP.
It includes approximate coverage for recurrent, scatter, RoIAlign, and
elementwise/reduction ops that are common in SGG predictors. Custom CUDA
kernels, NMS, sorting, indexing, Python control flow, and data loading are
still not included, so use the result as a consistent model-comparison estimate
rather than a hardware profiler result.
"""

import argparse
import copy
import logging
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CONFIG = "configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml"
FLOPS_PER_MAC = 1
FLOP_DEFINITION = "1 MAC = 1 FLOP"
GIGA_SCALE = 1e9
SUMMARY_KEYS = (
    "USE_GT_BOX",
    "USE_GT_OBJECT_LABEL",
    "PREDICT_USE_BIAS",
)
DETECTOR_COMPONENT_SCOPES = (
    ("backbone", "backbone"),
    ("neck", "neck"),
    ("rpn_head", "rpn_head"),
    ("detector_roi_head", "roi_head"),
    ("backbone_d2", "backbone_d2"),
    ("neck_d2", "neck_d2"),
    ("rpn_head_d2", "rpn_head_d2"),
    ("detector_roi_head_d2", "roi_head_d2"),
)

torch = None
nn = None
F = None
cfg = None
make_data_loader = None
build_detection_model = None


def macs_to_flops(macs):
    return int(macs) * FLOPS_PER_MAC


def flops_to_gflops(flops):
    return float(flops) / GIGA_SCALE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count approximate GFLOPs for the configured predcls model"
    )
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG,
        metavar="FILE",
        type=str,
        help="path to the maskrcnn_benchmark yaml config",
    )
    parser.add_argument(
        "--mm-config",
        "--mm_config",
        dest="mm_config",
        default=None,
        type=str,
        help="mmdet/mmrotate config used when cfg.Type is HBB/OBB",
    )
    parser.add_argument(
        "--mm-weight",
        "--mm_weight",
        dest="mm_weight",
        default=None,
        type=str,
        help="optional mmdet/mmrotate checkpoint used when cfg.Type is HBB/OBB",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=(
            "output directory or .txt file path. Defaults to "
            "cfg.OUTPUT_DIR/flops/model_gflops.txt"
        ),
    )
    parser.add_argument(
        "--num-batches",
        default=None,
        type=int,
        help=(
            "number of test batches to profile for quick debugging. "
            "Defaults to the full test set when omitted"
        ),
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def import_project_modules():
    global torch, nn, F, cfg, make_data_loader, build_detection_model

    import torch as torch_module
    import torch.nn as nn_module
    import torch.nn.functional as functional_module

    # Set up custom environment before importing most project modules.
    from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401
    from maskrcnn_benchmark.config import cfg as project_cfg
    from maskrcnn_benchmark.data import make_data_loader as project_make_data_loader
    from maskrcnn_benchmark.modeling.detector import (
        build_detection_model as project_build_detection_model,
    )

    torch = torch_module
    nn = nn_module
    F = functional_module
    cfg = project_cfg
    make_data_loader = project_make_data_loader
    build_detection_model = project_build_detection_model


def resolve_project_path(path_value):
    if path_value is None:
        return None

    path = Path(path_value)
    if path.exists():
        return path

    root_relative_path = ROOT_DIR / path
    if root_relative_path.exists():
        return root_relative_path

    return path


def resolve_output_path(output_arg, output_dir_from_cfg):
    if output_arg is None:
        output_dir = Path(output_dir_from_cfg) / "flops"
        return output_dir / "model_gflops.txt"

    output_path = Path(output_arg)
    if output_path.suffix.lower() == ".txt":
        return output_path
    if output_path.suffix:
        raise RuntimeError("--output must be a directory or .txt file path.")

    return output_path / "model_gflops.txt"


def make_logger():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("count_relation_gflops")


def get_device():
    device = torch.device(cfg.MODEL.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.MODEL.DEVICE is cuda, but torch.cuda.is_available() is False."
        )
    return device


def cfg_summary():
    relation_cfg = cfg.MODEL.ROI_RELATION_HEAD
    summary = {}
    for key in SUMMARY_KEYS:
        try:
            summary[key] = getattr(relation_cfg, key)
        except AttributeError:
            summary[key] = None
    return summary


def validate_predcls_config(args):
    if args.num_batches is not None and args.num_batches < 1:
        raise RuntimeError("--num-batches must be >= 1.")
    if not cfg.MODEL.RELATION_ON:
        raise RuntimeError("MODEL.RELATION_ON must be True for relation GFLOPs.")
    if not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        raise RuntimeError(
            "Only predcls is supported: MODEL.ROI_RELATION_HEAD.USE_GT_BOX "
            "must be True."
        )
    if not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
        raise RuntimeError(
            "Only predcls is supported: "
            "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL must be True."
        )

    requested_predictor = None
    opts = args.opts or []
    for key, value in zip(opts[0::2], opts[1::2]):
        if key == "MODEL.ROI_RELATION_HEAD.PREDICTOR":
            requested_predictor = value

    if (
        requested_predictor is not None
        and cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR != requested_predictor
    ):
        raise RuntimeError(
            "Resolved predictor does not match opts: "
            "{} != {}".format(
                cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR, requested_predictor
            )
        )


def build_mm_detector_and_data_cfg(args):
    if args.mm_config is None:
        raise RuntimeError("--mm-config is required when cfg.Type is HBB/OBB.")

    from mmcv import Config
    from mmcv.runner import load_checkpoint

    mm_config_path = resolve_project_path(args.mm_config)
    cfg_mmcv = Config.fromfile(str(mm_config_path))
    cfg_mmcv.model["ori_cfg"] = cfg

    if "OBB" in cfg.Type:
        from mmrotate.models import build_detector as build_mmrotate_detector

        model = build_mmrotate_detector(
            cfg_mmcv.model,
            train_cfg=cfg_mmcv.get("train_cfg"),
            test_cfg=cfg_mmcv.get("test_cfg"),
        )
    elif "HBB" in cfg.Type:
        from mmdet.models import build_detector as build_mmdet_detector

        model = build_mmdet_detector(
            cfg_mmcv.model,
            train_cfg=cfg_mmcv.get("train_cfg"),
            test_cfg=cfg_mmcv.get("test_cfg"),
        )
    else:
        raise RuntimeError("Unsupported cfg.Type for mm detector: {}".format(cfg.Type))

    model.to(get_device())

    if args.mm_weight:
        mm_weight_path = resolve_project_path(args.mm_weight)
        checkpoint = load_checkpoint(model, str(mm_weight_path), map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]

    data_cfg = copy.deepcopy(cfg)
    data_cfg["mmcv"] = cfg_mmcv.data.test
    return model, data_cfg


def build_profile_model_and_data_cfg(args):
    if cfg.Type == "CV":
        model = build_detection_model(cfg)
        model.to(get_device())
        return model, cfg
    if "OBB" in cfg.Type or "HBB" in cfg.Type:
        return build_mm_detector_and_data_cfg(args)
    raise RuntimeError("Unsupported cfg.Type for GFLOPs counting: {}".format(cfg.Type))


def get_relation_head(model):
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None or "relation" not in roi_heads:
        raise RuntimeError("model.roi_heads does not contain a relation head.")
    return roi_heads["relation"]


def get_predictor(model):
    relation_head = get_relation_head(model)
    predictor = getattr(relation_head, "predictor", None)
    if predictor is None:
        raise RuntimeError("model.roi_heads['relation'] does not contain predictor.")
    return predictor


def get_detector_scopes(model):
    scopes = {}
    for scope_name, attr_name in DETECTOR_COMPONENT_SCOPES:
        module = getattr(model, attr_name, None)
        if module is not None:
            scopes[scope_name] = module
    return scopes


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _to_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _linear_flops_from_tensors(input_tensor, weight_tensor, output_tensor):
    if (
        input_tensor is None
        or weight_tensor is None
        or output_tensor is None
        or input_tensor.dim() == 0
    ):
        return 0
    in_features = int(input_tensor.shape[-1])
    macs = int(output_tensor.numel()) * in_features
    return macs_to_flops(macs)


def _matmul_flops(a_tensor, b_tensor, output_tensor):
    if (
        a_tensor is None
        or b_tensor is None
        or output_tensor is None
        or not torch.is_tensor(a_tensor)
        or not torch.is_tensor(b_tensor)
        or a_tensor.dim() == 0
        or b_tensor.dim() == 0
    ):
        return 0

    if a_tensor.dim() == 1 and b_tensor.dim() == 1:
        macs = int(a_tensor.shape[0])
    elif a_tensor.dim() == 2 and b_tensor.dim() == 1:
        macs = int(a_tensor.shape[0]) * int(a_tensor.shape[1])
    elif a_tensor.dim() == 1 and b_tensor.dim() == 2:
        macs = int(a_tensor.shape[0]) * int(b_tensor.shape[-1])
    else:
        macs = int(output_tensor.numel()) * int(a_tensor.shape[-1])
    return macs_to_flops(macs)


def _tensor_numel(value):
    if torch.is_tensor(value):
        return int(value.numel())
    tensor = _first_tensor(value)
    return 0 if tensor is None else int(tensor.numel())


def _primary_arg(args):
    if isinstance(args, (list, tuple)) and args:
        return args[0]
    return args


def _is_packed_sequence_like(value):
    return hasattr(value, "data") and hasattr(value, "batch_sizes")


def _sequence_element_count(value, batch_first=False):
    value = _primary_arg(value)
    if _is_packed_sequence_like(value) and torch.is_tensor(value.data):
        return int(value.data.shape[0])
    if torch.is_tensor(value):
        if value.dim() >= 3:
            if batch_first:
                return int(value.shape[0]) * int(value.shape[1])
            return int(value.shape[0]) * int(value.shape[1])
        if value.dim() >= 2:
            return int(value.shape[0])
    return 0


def _cell_batch_size(inputs):
    tensor = _first_tensor(inputs)
    if tensor is None:
        return 0
    if tensor.dim() == 0:
        return 1
    return int(tensor.shape[0])


def _recurrent_module_flops(module, inputs):
    if isinstance(module, nn.LSTM):
        gates = 4
        op_name = "lstm"
    elif isinstance(module, nn.GRU):
        gates = 3
        op_name = "gru"
    elif isinstance(module, nn.RNN):
        gates = 1
        op_name = "rnn"
    else:
        return 0, None

    total_steps = _sequence_element_count(
        inputs, batch_first=bool(getattr(module, "batch_first", False))
    )
    if total_steps < 1:
        return 0, None

    hidden_size = int(module.hidden_size)
    num_layers = int(module.num_layers)
    num_directions = 2 if bool(module.bidirectional) else 1
    macs = 0
    for layer in range(num_layers):
        if layer == 0:
            layer_input_size = int(module.input_size)
        else:
            layer_input_size = hidden_size * num_directions
        per_direction_macs = total_steps * gates * (
            layer_input_size * hidden_size + hidden_size * hidden_size
        )
        macs += per_direction_macs * num_directions
    return macs_to_flops(macs), op_name


def _recurrent_cell_flops(module, inputs):
    if isinstance(module, nn.LSTMCell):
        gates = 4
        op_name = "lstm_cell"
    elif isinstance(module, nn.GRUCell):
        gates = 3
        op_name = "gru_cell"
    elif isinstance(module, nn.RNNCell):
        gates = 1
        op_name = "rnn_cell"
    else:
        return 0, None

    batch_size = _cell_batch_size(inputs)
    if batch_size < 1:
        return 0, None
    input_size = int(module.input_size)
    hidden_size = int(module.hidden_size)
    macs = batch_size * gates * (input_size * hidden_size + hidden_size * hidden_size)
    return macs_to_flops(macs), op_name


def _roi_align_flops(module, output):
    output_tensor = _first_tensor(output)
    if output_tensor is None:
        return 0
    sampling_ratio = int(getattr(module, "sampling_ratio", 0) or 0)
    samples = sampling_ratio * sampling_ratio if sampling_ratio > 0 else 1
    # Bilinear interpolation is approximated as four weighted samples per bin.
    return int(output_tensor.numel()) * max(samples, 1) * 4


def _count_trainable_params(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class FlopCounter(object):
    def __init__(self, model, scopes):
        self.model = model
        self.scopes = scopes
        self.scope_module_ids = {
            name: set(id(module) for module in scope.modules())
            for name, scope in scopes.items()
        }
        self.direct_trainable_param_ids = {
            id(module): set(
                id(param)
                for _, param in module.named_parameters(recurse=False)
                if param.requires_grad
            )
            for module in model.modules()
        }
        self.stack = []
        self.handles = []
        self.originals = []
        self._suppress_addmm_depth = 0
        self.flops = defaultdict(int)
        self.flops_by_op = defaultdict(lambda: defaultdict(int))
        self.executed_trainable_param_ids = defaultdict(set)

    def __enter__(self):
        self.reset()
        self._register_module_hooks()
        self._patch_torch_functions()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._restore_torch_functions()
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.stack = []
        self._suppress_addmm_depth = 0

    def reset(self):
        self.flops = defaultdict(int)
        self.flops_by_op = defaultdict(lambda: defaultdict(int))
        self.executed_trainable_param_ids = defaultdict(set)
        self._suppress_addmm_depth = 0

    def _register_module_hooks(self):
        for module in self.model.modules():
            self.handles.append(module.register_forward_pre_hook(self._pre_hook))
            self.handles.append(module.register_forward_hook(self._post_hook))

    def _pre_hook(self, module, inputs):
        self.stack.append(module)
        self._record_executed_trainable_params(module)

    def _post_hook(self, module, inputs, output):
        try:
            flops, op_name = self._module_flops(module, inputs, output)
            if flops:
                self._add_flops_for_module(module, op_name, flops)
        finally:
            if self.stack:
                self.stack.pop()

    def _module_flops(self, module, inputs, output):
        input_tensor = _first_tensor(inputs)
        flops, op_name = _recurrent_module_flops(module, inputs)
        if flops:
            return flops, op_name
        flops, op_name = _recurrent_cell_flops(module, inputs)
        if flops:
            return flops, op_name
        if module.__class__.__name__ == "ROIAlign":
            flops = _roi_align_flops(module, output)
            if flops:
                return flops, "roi_align_approx"

        output_tensor = _first_tensor(output)
        if output_tensor is None:
            return 0, None

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel_ops = _prod(_to_tuple(module.kernel_size))
            kernel_ops *= int(module.in_channels // module.groups)
            macs = int(output_tensor.numel()) * kernel_ops
            return macs_to_flops(macs), "conv"

        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if input_tensor is None:
                return 0, None
            kernel_ops = _prod(_to_tuple(module.kernel_size))
            kernel_ops *= int(module.out_channels // module.groups)
            macs = int(input_tensor.numel()) * kernel_ops
            return macs_to_flops(macs), "conv_transpose"

        if isinstance(module, nn.Linear):
            return (
                _linear_flops_from_tensors(
                    input_tensor, getattr(module, "weight", None), output_tensor
                ),
                "linear",
            )

        norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )
        if isinstance(module, norm_types):
            return int(output_tensor.numel()) * 2, "normalization"

        return 0, None

    def _current_module(self):
        return self.stack[-1] if self.stack else None

    def _recurrent_types(self):
        return (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    def _inside_module_types(self, module_types):
        return any(isinstance(module, module_types) for module in self.stack)

    def _inside_counted_linear_or_recurrent(self):
        return self._inside_module_types((nn.Linear,) + self._recurrent_types())

    def _active_scope_names(self, module):
        if module is None:
            return ("total",)
        module_id = id(module)
        names = []
        for scope_name, module_ids in self.scope_module_ids.items():
            if module_id in module_ids:
                names.append(scope_name)
        return names

    def _add_flops_for_module(self, module, op_name, flops):
        if not op_name or flops <= 0:
            return
        for scope_name in self._active_scope_names(module):
            self.flops[scope_name] += int(flops)
            self.flops_by_op[scope_name][op_name] += int(flops)

    def _add_flops_for_current_module(self, op_name, flops):
        self._add_flops_for_module(self._current_module(), op_name, flops)

    def _record_executed_trainable_params(self, module):
        param_ids = self.direct_trainable_param_ids.get(id(module), set())
        if not param_ids:
            return
        for scope_name in self._active_scope_names(module):
            self.executed_trainable_param_ids[scope_name].update(param_ids)

    def _patch(self, owner, name, replacement):
        try:
            original = getattr(owner, name)
            setattr(owner, name, replacement(original))
            self.originals.append((owner, name, original))
        except Exception:
            return

    def _patch_torch_functions(self):
        self._patch(F, "linear", self._wrap_functional_linear)
        self._patch(torch, "matmul", self._wrap_matmul)
        self._patch(torch, "mm", self._wrap_matmul)
        self._patch(torch, "bmm", self._wrap_matmul)
        self._patch(torch, "addmm", self._wrap_addmm)
        self._patch(torch.Tensor, "matmul", self._wrap_tensor_matmul)
        self._patch(torch.Tensor, "mm", self._wrap_tensor_matmul)
        self._patch(torch.Tensor, "bmm", self._wrap_tensor_matmul)
        self._patch(torch.Tensor, "__matmul__", self._wrap_tensor_dunder_matmul)
        self._patch(torch.Tensor, "__rmatmul__", self._wrap_tensor_dunder_rmatmul)
        self._patch_elementwise_functions()
        self._patch_torch_scatter_functions()

    def _restore_torch_functions(self):
        while self.originals:
            owner, name, original = self.originals.pop()
            try:
                setattr(owner, name, original)
            except Exception:
                pass

    def _wrap_functional_linear(self, original):
        def wrapped(input_tensor, weight, bias=None):
            should_count = not self._inside_counted_linear_or_recurrent()
            if should_count:
                self._suppress_addmm_depth += 1
            try:
                result = original(input_tensor, weight, bias)
            finally:
                if should_count:
                    self._suppress_addmm_depth -= 1
            if should_count:
                flops = _linear_flops_from_tensors(input_tensor, weight, result)
                self._add_flops_for_current_module("functional_linear", flops)
            return result

        return wrapped

    def _wrap_matmul(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(args) >= 2 and torch.is_tensor(result):
                flops = _matmul_flops(args[0], args[1], result)
                self._add_flops_for_current_module("matmul", flops)
            return result

        return wrapped

    def _wrap_tensor_matmul(self, original):
        def wrapped(tensor, other, *args, **kwargs):
            result = original(tensor, other, *args, **kwargs)
            if torch.is_tensor(result):
                flops = _matmul_flops(tensor, other, result)
                self._add_flops_for_current_module("matmul", flops)
            return result

        return wrapped

    def _wrap_tensor_dunder_matmul(self, original):
        def wrapped(tensor, other):
            result = original(tensor, other)
            if torch.is_tensor(result):
                flops = _matmul_flops(tensor, other, result)
                self._add_flops_for_current_module("tensor_matmul", flops)
            return result

        return wrapped

    def _wrap_tensor_dunder_rmatmul(self, original):
        def wrapped(tensor, other):
            result = original(tensor, other)
            if torch.is_tensor(result):
                flops = _matmul_flops(other, tensor, result)
                self._add_flops_for_current_module("tensor_matmul", flops)
            return result

        return wrapped

    def _wrap_addmm(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if (
                self._suppress_addmm_depth > 0
                or self._inside_counted_linear_or_recurrent()
            ):
                return result

            mat1 = args[1] if len(args) >= 2 else kwargs.get("mat1")
            mat2 = args[2] if len(args) >= 3 else kwargs.get("mat2")
            if mat1 is not None and mat2 is not None and torch.is_tensor(result):
                flops = _matmul_flops(mat1, mat2, result)
                self._add_flops_for_current_module("addmm", flops)
            return result

        return wrapped

    def _patch_elementwise_functions(self):
        for owner, names in (
            (F, ("softmax", "relu", "elu", "leaky_relu")),
            (
                torch,
                ("softmax", "sigmoid", "tanh", "relu", "exp", "norm", "cdist", "mul"),
            ),
            (
                torch.Tensor,
                ("softmax", "sigmoid", "tanh", "relu", "exp", "norm", "mul"),
            ),
        ):
            for name in names:
                if hasattr(owner, name):
                    self._patch(owner, name, self._wrap_elementwise_function(name))

    def _wrap_elementwise_function(self, name):
        def replacement(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                flops = self._elementwise_flops(name, args, kwargs, result)
                self._add_flops_for_current_module("elementwise_approx", flops)
                return result

            return wrapped

        return replacement

    def _elementwise_flops(self, name, args, kwargs, result):
        result_tensor = _first_tensor(result)
        if name == "cdist" and len(args) >= 2 and result_tensor is not None:
            a_tensor, b_tensor = args[0], args[1]
            if torch.is_tensor(a_tensor) and torch.is_tensor(b_tensor):
                return int(result_tensor.numel()) * int(a_tensor.shape[-1]) * 3
        if name in ("norm",):
            input_tensor = _first_tensor(args)
            return 0 if input_tensor is None else int(input_tensor.numel()) * 3
        if result_tensor is None:
            return 0
        if name == "softmax":
            return int(result_tensor.numel()) * 5
        return int(result_tensor.numel())

    def _patch_torch_scatter_functions(self):
        try:
            import torch_scatter as torch_scatter_module
        except Exception:
            return

        originals = {}
        for name in ("scatter", "scatter_add", "segment_csr", "gather_csr"):
            if hasattr(torch_scatter_module, name):
                originals[name] = getattr(torch_scatter_module, name)

        for module in list(sys.modules.values()):
            if module is None:
                continue
            for name, original in originals.items():
                try:
                    if getattr(module, name, None) is original:
                        self._patch(module, name, self._wrap_torch_scatter(name))
                except Exception:
                    continue

    def _wrap_torch_scatter(self, name):
        def replacement(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                flops = self._torch_scatter_flops(name, args, kwargs, result)
                self._add_flops_for_current_module("scatter_reduce", flops)
                return result

            return wrapped

        return replacement

    def _torch_scatter_flops(self, name, args, kwargs, result):
        if name in ("scatter", "scatter_add", "segment_csr"):
            return _tensor_numel(args[0]) if args else 0
        if name == "gather_csr":
            return _tensor_numel(result)
        return 0

    def summary(self):
        result = {}
        for scope_name in self.scopes:
            flops = int(self.flops.get(scope_name, 0))
            result[scope_name] = {
                "flops": flops,
                "gflops": flops_to_gflops(flops),
                "executed_trainable_param_count": len(
                    self.executed_trainable_param_ids.get(scope_name, set())
                ),
            }
        return result


def unpack_batch(batch):
    if len(batch) >= 5:
        images, targets, image_ids, imgs, tar1 = batch[:5]
    elif len(batch) >= 3:
        images, targets, image_ids = batch[:3]
        imgs, tar1 = None, None
    else:
        raise RuntimeError("Unexpected batch format with {} entries.".format(len(batch)))
    return images, targets, image_ids, imgs, tar1


def move_targets_to_device(targets, device):
    return [target.to(device) for target in targets]


def image_shape(images):
    if hasattr(images, "tensors"):
        return [int(dim) for dim in images.tensors.shape]
    if torch.is_tensor(images):
        return [int(dim) for dim in images.shape]
    return None


def image_sizes(images):
    if not hasattr(images, "image_sizes"):
        return None
    return [[int(dim) for dim in size] for size in images.image_sizes]


def _boxlist_scalar(boxlist, field):
    if not hasattr(boxlist, "has_field") or not boxlist.has_field(field):
        return None
    value = boxlist.get_field(field)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.detach().view(-1)[0].cpu().item()
    return int(value)


def collect_object_counts(outputs, targets):
    return [int(len(target)) for target in targets]


def collect_relation_pair_counts(outputs, targets):
    counts = []
    if isinstance(outputs, (list, tuple)):
        for output, target in zip(outputs, targets):
            if hasattr(output, "has_field") and output.has_field("rel_pair_idxs"):
                counts.append(int(output.get_field("rel_pair_idxs").shape[0]))
            else:
                num_objects = len(target)
                counts.append(int(num_objects * max(num_objects - 1, 0)))
        if len(counts) < len(targets):
            counts.extend(
                int(len(target) * max(len(target) - 1, 0))
                for target in targets[len(counts):]
            )
        return counts
    return [int(len(target) * max(len(target) - 1, 0)) for target in targets]


def build_batch_report(
    dataset_name,
    dataset_index,
    batch_index,
    dataset_batch_index,
    images,
    targets,
    outputs,
    counter_summary,
):
    num_objects = collect_object_counts(outputs, targets)
    num_relation_pairs = collect_relation_pair_counts(outputs, targets)
    return {
        "dataset_name": dataset_name,
        "dataset_index": int(dataset_index),
        "batch_index": int(batch_index),
        "dataset_batch_index": int(dataset_batch_index),
        "image_shape": image_shape(images),
        "image_sizes": image_sizes(images),
        "num_images": int(len(targets)),
        "image_ids": [],
        "num_objects": num_objects,
        "num_relation_pairs": num_relation_pairs,
        "total_flops": int(counter_summary["total"]["flops"]),
        "relation_head_flops": int(counter_summary["relation_head"]["flops"]),
        "predictor_flops": int(counter_summary["predictor"]["flops"]),
    }


def profile_batches(model, data_cfg, args, logger):
    device = get_device()
    data_loaders = make_data_loader(data_cfg, mode="test", is_distributed=False)
    if not data_loaders:
        raise RuntimeError("make_data_loader returned no test dataloaders.")
    dataset_names = list(data_cfg.DATASETS.TEST)
    if len(dataset_names) != len(data_loaders):
        raise RuntimeError(
            "DATASETS.TEST has {} entries but make_data_loader returned {} loaders.".format(
                len(dataset_names), len(data_loaders)
            )
        )

    relation_head = get_relation_head(model)
    predictor = get_predictor(model)
    scopes = {
        "total": model,
        "relation_head": relation_head,
        "predictor": predictor,
    }
    scopes.update(get_detector_scopes(model))

    model.eval()
    batch_reports = []
    executed_param_ids_by_scope = defaultdict(set)
    global_batch_index = 0

    for dataset_index, (dataset_name, data_loader) in enumerate(
        zip(dataset_names, data_loaders)
    ):
        for dataset_batch_index, batch in enumerate(data_loader):
            if (
                args.num_batches is not None
                and global_batch_index >= args.num_batches
            ):
                break

            images, targets, image_ids, imgs, tar1 = unpack_batch(batch)
            targets = move_targets_to_device(targets, device)
            images = images.to(device)
            sgd_data = [imgs, tar1] if imgs is not None else None

            with FlopCounter(model, scopes) as counter:
                with torch.no_grad():
                    outputs = model(images, targets, logger=logger, sgd_data=sgd_data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

            counter_summary = counter.summary()
            for scope_name, param_ids in counter.executed_trainable_param_ids.items():
                executed_param_ids_by_scope[scope_name].update(param_ids)
            batch_report = build_batch_report(
                dataset_name,
                dataset_index,
                global_batch_index,
                dataset_batch_index,
                images,
                targets,
                outputs,
                counter_summary,
            )
            batch_report["image_ids"] = [int(image_id) for image_id in image_ids]
            batch_reports.append(batch_report)
            global_batch_index += 1

        if args.num_batches is not None and global_batch_index >= args.num_batches:
            break

    if not batch_reports:
        raise RuntimeError("No batches were profiled.")

    return batch_reports, executed_param_ids_by_scope


def _sum(batch_reports, key):
    return sum(float(report[key]) for report in batch_reports)


def _sum_images(batch_reports):
    return sum(int(report["num_images"]) for report in batch_reports)


def _sum_list_values(batch_reports, key):
    return sum(sum(int(value) for value in report[key]) for report in batch_reports)


def make_report(args, model, batch_reports, executed_param_ids_by_scope=None):
    executed_param_ids_by_scope = executed_param_ids_by_scope or {}
    predictor = get_predictor(model)
    num_images_profiled = _sum_images(batch_reports)
    if num_images_profiled < 1:
        raise RuntimeError("No images were profiled.")

    profiled_total_flops = _sum(batch_reports, "total_flops")
    profiled_relation_head_flops = _sum(batch_reports, "relation_head_flops")
    profiled_predictor_flops = _sum(batch_reports, "predictor_flops")

    avg_total_flops_per_image = profiled_total_flops / float(num_images_profiled)
    avg_relation_head_flops_per_image = (
        profiled_relation_head_flops / float(num_images_profiled)
    )
    avg_predictor_flops_per_image = (
        profiled_predictor_flops / float(num_images_profiled)
    )
    avg_relation_feature_flops_per_image = (
        avg_relation_head_flops_per_image - avg_predictor_flops_per_image
    )
    total_num_objects_profiled = _sum_list_values(batch_reports, "num_objects")
    total_num_relation_pairs_profiled = _sum_list_values(
        batch_reports, "num_relation_pairs"
    )
    avg_num_objects_per_image = total_num_objects_profiled / float(num_images_profiled)
    avg_num_relation_pairs_per_image = (
        total_num_relation_pairs_profiled / float(num_images_profiled)
    )

    predictor_trainable_params = _count_trainable_params(predictor)

    report = {
        "flop_definition": FLOP_DEFINITION,
        "Type": cfg.Type,
        "predictor_cfg": cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR,
        "predictor_class": predictor.__class__.__name__,
        "num_batches_profiled": int(len(batch_reports)),
        "num_images_profiled": int(num_images_profiled),
        "avg_num_objects_per_image": float(avg_num_objects_per_image),
        "avg_num_relation_pairs_per_image": float(avg_num_relation_pairs_per_image),
        "predictor_trainable_params_M": float(predictor_trainable_params / 1e6),
        "avg_total_gflops_per_image": flops_to_gflops(avg_total_flops_per_image),
        "avg_relation_head_gflops_per_image": flops_to_gflops(
            avg_relation_head_flops_per_image
        ),
        "avg_predictor_gflops_per_image": flops_to_gflops(
            avg_predictor_flops_per_image
        ),
        "avg_relation_feature_gflops_per_image": flops_to_gflops(
            avg_relation_feature_flops_per_image
        ),
    }
    report.update(cfg_summary())
    return report


def report_text_lines(report, txt_path=None, leading_blank=False):
    lines = []
    if leading_blank:
        lines.append("")

    lines.append("Predcls GFLOPs count")
    lines.append("-" * 40)
    report_groups = (
        (
            "task",
            (
                "Type",
                "predictor_cfg",
                "predictor_class",
                "USE_GT_BOX",
                "USE_GT_OBJECT_LABEL",
                "PREDICT_USE_BIAS",
            ),
        ),
        (
            "profile",
            (
                "num_images_profiled",
                "num_batches_profiled",
                "avg_num_objects_per_image",
                "avg_num_relation_pairs_per_image",
            ),
        ),
        (
            "params_M",
            (
                "predictor_trainable_params_M",
            ),
        ),
        (
            "gflops_per_image",
            (
                "avg_total_gflops_per_image",
                "avg_relation_head_gflops_per_image",
                "avg_predictor_gflops_per_image",
                "avg_relation_feature_gflops_per_image",
            ),
        ),
    )
    for group_name, keys in report_groups:
        values = ["{}={}".format(key, report.get(key)) for key in keys]
        lines.append("{}: {}".format(group_name, ", ".join(values)))
    lines.append("flop_definition: {}".format(report.get("flop_definition")))
    lines.append("-" * 40)
    if txt_path is not None:
        lines.append("txt:  {}".format(txt_path))
    return lines


def write_txt(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_text_lines(report, path)))
        f.write("\n")


def print_report(report, txt_path):
    print("\n".join(report_text_lines(report, txt_path, True)))


def main():
    args = parse_args()
    import_project_modules()

    args.config_file = resolve_project_path(args.config_file)
    if args.mm_config is not None:
        args.mm_config = resolve_project_path(args.mm_config)
    if args.mm_weight is not None:
        args.mm_weight = resolve_project_path(args.mm_weight)

    cfg.merge_from_file(str(args.config_file))
    cfg.merge_from_list(args.opts or [])
    validate_predcls_config(args)

    logger = make_logger()
    model, data_cfg = build_profile_model_and_data_cfg(args)
    batch_reports, executed_param_ids_by_scope = profile_batches(
        model, data_cfg, args, logger
    )
    report = make_report(args, model, batch_reports, executed_param_ids_by_scope)

    txt_path = resolve_output_path(args.output, cfg.OUTPUT_DIR)
    write_txt(txt_path, report)
    print_report(report, txt_path)


if __name__ == "__main__":
    main()
