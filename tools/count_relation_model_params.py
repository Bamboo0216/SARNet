# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Count model-level parameters for the active scene graph relation predictor.

This script intentionally follows the same config precedence used by the
training and test scripts:

    defaults.py -> --config-file -> opts

It reports only the currently configured relation predictor:

    model.roi_heads["relation"].predictor
"""

import argparse
import copy
import logging
import sys
import traceback
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Set up custom environment before importing most project modules.
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model


DEFAULT_CONFIG = "configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count parameters of the configured relation predictor"
    )
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output directory or .txt file path. Defaults to "
            "cfg.OUTPUT_DIR/param_count/model_params.txt"
        ),
        type=str,
    )
    parser.add_argument(
        "--used-mode",
        default="none",
        choices=("none", "backward"),
        help="whether to estimate actually used predictor parameters with backward",
    )
    parser.add_argument(
        "--num-used-batches",
        default=1,
        type=int,
        help="number of train batches used for backward-usage coverage",
    )
    parser.add_argument(
        "--mm-config",
        default=None,
        type=str,
        help="mmdet/mmrotate config used for HBB/OBB backward mode",
    )
    parser.add_argument(
        "--mm-weight",
        default=None,
        type=str,
        help="optional detector checkpoint used for HBB/OBB backward mode",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def count_parameters(module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params


def resolve_project_path(path_value):
    if path_value is None:
        return None

    config_path = Path(path_value)
    if config_path.exists():
        return config_path

    root_relative_path = ROOT_DIR / config_path
    if root_relative_path.exists():
        return root_relative_path

    return config_path


def resolve_config_file(config_file):
    return resolve_project_path(config_file)


def resolve_output_path(output_arg, output_dir_from_cfg):
    if output_arg is None:
        output_dir = Path(output_dir_from_cfg) / "param_count"
        return output_dir / "model_params.txt"

    output_path = Path(output_arg)
    if output_path.suffix.lower() == ".txt":
        return output_path
    if output_path.suffix:
        raise RuntimeError("--output must be a directory or .txt file path.")

    return output_path / "model_params.txt"


def ensure_output_dir():
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def make_logger():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("count_relation_model_params")


def get_device():
    return torch.device(cfg.MODEL.DEVICE)


def get_predictor(model):
    if model.roi_heads is None or "relation" not in model.roi_heads:
        raise RuntimeError("model.roi_heads does not contain a relation head.")
    return model.roi_heads["relation"].predictor


def make_report(predictor, total_params, trainable_params):
    return {
        "predictor_cfg": cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR,
        "predictor_class": predictor.__class__.__name__,
        "total_params": int(total_params),
        "total_params_M": float(total_params / 1e6),
        "trainable_params_M": float(trainable_params / 1e6),
        "used_param_status": "not_requested",
        "used_param_error": None,
        "used_params_M": None,
        "used_param_ratio": None,
    }


def update_report_with_used_params(report, used_params):
    report.update(
        {
            "used_param_status": "ok",
            "used_param_error": None,
            "used_params_M": float(used_params / 1e6),
            "used_param_ratio": float(used_params / report["total_params"])
            if report["total_params"]
            else None,
        }
    )


def update_report_with_used_param_failure(report, exc):
    report.update(
        {
            "used_param_status": "failed",
            "used_param_error": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "used_params_M": None,
            "used_param_ratio": None,
        }
    )


def report_text_lines(report, txt_path=None, leading_blank=False):
    lines = []
    if leading_blank:
        lines.append("")

    lines.append("Relation predictor parameter count")
    lines.append("-" * 40)
    for key in (
        "predictor_cfg",
        "predictor_class",
        "total_params_M",
        "trainable_params_M",
        "used_param_status",
        "used_params_M",
        "used_param_ratio",
    ):
        lines.append(f"{key}: {report[key]}")
    if report.get("used_param_status") == "failed" and report.get("used_param_error"):
        error_lines = report["used_param_error"].strip().splitlines()
        lines.append("used_param_error_tail:")
        for line in error_lines[-8:]:
            lines.append(line)
    lines.append("-" * 40)
    if txt_path is not None:
        lines.append(f"txt: {txt_path}")
    return lines


def write_txt(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_text_lines(report, path)))
        f.write("\n")


def print_report(report, txt_path):
    print("\n".join(report_text_lines(report, txt_path, True)))


def freeze_module_params(modules):
    for module in modules:
        if module is None:
            continue
        for _, param in module.named_parameters():
            param.requires_grad = False


def modules_to_freeze(model):
    if cfg.Type == "CV":
        return (
            getattr(model, "rpn", None),
            getattr(model, "backbone", None),
            getattr(getattr(model, "roi_heads", None), "box", None),
        )

    return (
        getattr(model, "neck", None),
        getattr(model, "backbone", None),
        getattr(model, "rpn_head", None),
        getattr(model, "roi_head", None),
    )


def build_mm_detector(args):
    if args.mm_config is None:
        raise RuntimeError(
            "--mm-config is required for --used-mode backward when cfg.Type is HBB/OBB."
        )

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
        raise RuntimeError(f"Unsupported non-CV cfg.Type for mm detector: {cfg.Type}")

    model.to(get_device())

    if args.mm_weight:
        mm_weight_path = resolve_project_path(args.mm_weight)
        checkpoint = load_checkpoint(model, str(mm_weight_path), map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]

    cfg_train = copy.deepcopy(cfg)
    cfg_train["mmcv"] = cfg_mmcv.data.train
    return model, cfg_train


def build_count_model(args):
    if args.used_mode == "backward" and cfg.Type != "CV":
        return build_mm_detector(args)

    model = build_detection_model(cfg)
    if args.used_mode == "backward":
        model.to(get_device())
    return model, cfg


def zero_model_grads(model):
    for param in model.parameters():
        param.grad = None


def unpack_train_batch(batch):
    if len(batch) >= 5:
        images, targets, image_ids, imgs, tar1 = batch[:5]
    elif len(batch) >= 3:
        images, targets, image_ids = batch[:3]
        imgs, tar1 = None, None
    else:
        raise RuntimeError(f"Unexpected train batch format with {len(batch)} entries.")
    return images, targets, image_ids, imgs, tar1


def move_targets_to_device(targets, device):
    return [target.to(device) for target in targets]


def iter_loss_tensors(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            yield value
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("__"):
                continue
            yield from iter_loss_tensors(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_loss_tensors(item)


def sum_loss_tensors(loss_dict):
    loss_tensors = list(iter_loss_tensors(loss_dict))
    if not loss_tensors:
        raise RuntimeError("Model forward did not return any floating point tensor loss.")

    total_loss = loss_tensors[0]
    for loss in loss_tensors[1:]:
        total_loss = total_loss + loss

    if not total_loss.requires_grad:
        raise RuntimeError("The summed loss does not require gradients.")
    return total_loss


def count_used_params_by_backward(model, predictor, train_cfg, args, logger):
    train_data_loader = make_data_loader(
        train_cfg,
        mode="train",
        is_distributed=False,
        start_iter=0,
    )
    train_iter = iter(train_data_loader)
    device = get_device()
    used_param_names = set()
    successful_batches = 0
    attempts = 0
    max_attempts = max(args.num_used_batches * 20, 20)

    while successful_batches < args.num_used_batches and attempts < max_attempts:
        attempts += 1
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data_loader)
            batch = next(train_iter)

        images, targets, _, imgs, tar1 = unpack_train_batch(batch)
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        sgd_data = [imgs, tar1] if imgs is not None else None

        model.train()
        freeze_module_params(modules_to_freeze(model))
        zero_model_grads(model)

        loss_dict = model(
            images,
            targets,
            ite=successful_batches + 1,
            logger=logger,
            sgd_data=sgd_data,
        )
        if isinstance(loss_dict, dict) and loss_dict.get("__skip_batch__", False):
            continue

        losses = sum_loss_tensors(loss_dict)
        losses.backward()

        for name, param in predictor.named_parameters():
            if param.grad is not None:
                used_param_names.add(name)

        successful_batches += 1

    if successful_batches < args.num_used_batches:
        raise RuntimeError(
            "Could not collect enough usable training batches: "
            f"{successful_batches}/{args.num_used_batches}."
        )

    used_params = sum(
        param.numel()
        for name, param in predictor.named_parameters()
        if name in used_param_names
    )
    return used_params, successful_batches


def main():
    args = parse_args()
    opts = args.opts or []
    args.config_file = resolve_config_file(args.config_file)

    cfg.merge_from_file(str(args.config_file))
    cfg.merge_from_list(opts)

    if args.used_mode == "backward" and args.num_used_batches < 1:
        raise RuntimeError("--num-used-batches must be >= 1 for backward mode.")

    if not cfg.MODEL.RELATION_ON:
        raise RuntimeError(
            "MODEL.RELATION_ON is False, so model.roi_heads['relation'] will not exist."
        )

    requested_predictor = None
    for key, value in zip(opts[0::2], opts[1::2]):
        if key == "MODEL.ROI_RELATION_HEAD.PREDICTOR":
            requested_predictor = value

    if (
        requested_predictor is not None
        and cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR != requested_predictor
    ):
        raise RuntimeError(
            "Resolved predictor does not match opts: "
            f"{cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR} != {requested_predictor}"
        )

    ensure_output_dir()
    logger = make_logger()
    try:
        model, train_cfg = build_count_model(args)
        predictor = get_predictor(model)
        total_params, trainable_params = count_parameters(predictor)
        report = make_report(predictor, total_params, trainable_params)
    except Exception as exc:
        if args.used_mode != "backward":
            raise
        static_model = build_detection_model(cfg)
        predictor = get_predictor(static_model)
        total_params, trainable_params = count_parameters(predictor)
        report = make_report(predictor, total_params, trainable_params)
        update_report_with_used_param_failure(report, exc)
        txt_path = resolve_output_path(args.output, cfg.OUTPUT_DIR)
        write_txt(txt_path, report)
        print_report(report, txt_path)
        return

    if args.used_mode == "backward":
        try:
            used_params, _ = count_used_params_by_backward(
                model, predictor, train_cfg, args, logger
            )
            update_report_with_used_params(report, used_params)
        except Exception as exc:  # keep total params even if runtime probing fails
            update_report_with_used_param_failure(report, exc)

    txt_path = resolve_output_path(args.output, cfg.OUTPUT_DIR)
    write_txt(txt_path, report)
    print_report(report, txt_path)


if __name__ == "__main__":
    main()
