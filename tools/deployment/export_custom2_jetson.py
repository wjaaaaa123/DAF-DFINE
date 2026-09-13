#!/usr/bin/env python3
"""Export the actual custom2 EMA-3 checkpoint for Jetson TensorRT 8.x."""

import argparse
from pathlib import Path
import sys

import onnx
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/ultralytics-yaml/deploy_deim_hgnetv2_s_custom2.yml"
DEFAULT_CHECKPOINT = ROOT / "outputs/deim_hgnetv2_s_custom2/best_stg2.pth"
DEFAULT_OUTPUT = ROOT / "jetson_single_image/model.onnx"
DEFAULT_LABELS = ROOT / "jetson_single_image/labels.txt"


class DeployModel(nn.Module):
    def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


class MeanPoolHeight(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=3, keepdim=True)


class MeanPoolWidth(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=2, keepdim=True)


class MeanPoolSpatial(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=(2, 3), keepdim=True)


def replace_ema_pooling_for_onnx(model: nn.Module) -> int:
    """Replace EMA adaptive pools with mathematically identical ReduceMean ops."""
    from engine.extre_module.custom_nn.attention.ema import EMA

    replaced = 0
    for module in model.modules():
        if isinstance(module, EMA):
            module.pool_h = MeanPoolHeight()
            module.pool_w = MeanPoolWidth()
            module.agp = MeanPoolSpatial()
            replaced += 1
    return replaced


def load_class_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if len(names) != 10:
        raise ValueError(f"类别文件必须包含 10 行，实际为 {len(names)} 行: {path}")
    return names


def build_model(config_path: Path, checkpoint_path: Path) -> DeployModel:
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg.setdefault("DEIM_MG", {})["pretrained"] = False

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        state_source = "ema.module"
    elif "model" in checkpoint:
        state = checkpoint["model"]
        state_source = "model"
    else:
        raise KeyError("权重中既没有 ema.module，也没有 model")

    cfg.model.load_state_dict(state, strict=True)
    print(f"严格加载权重成功: {state_source}")
    replaced = replace_ema_pooling_for_onnx(cfg.model)
    if replaced != 4:
        raise RuntimeError(f"预期替换 4 个 EMA 注意力层，实际替换 {replaced} 个")
    print(f"已将 {replaced} 个 EMA 池化层替换为 ONNX ReduceMean")
    return DeployModel(cfg.model, cfg.postprocessor).eval()


def export_model(
    model: DeployModel,
    output_path: Path,
    class_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = torch.rand(1, 3, 640, 640, dtype=torch.float32)
    orig_target_sizes = torch.tensor([[640.0, 640.0]], dtype=torch.float32)

    with torch.inference_mode():
        outputs = model(images, orig_target_sizes)
    if len(outputs) != 3:
        raise RuntimeError(f"部署模型应输出 3 项，实际输出 {len(outputs)} 项")
    print(
        "PyTorch 输出形状: "
        + ", ".join(str(tuple(value.shape)) for value in outputs)
    )

    torch.onnx.export(
        model,
        (images, orig_target_sizes),
        str(output_path),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    del onnx_model.metadata_props[:]
    metadata = onnx_model.metadata_props.add()
    metadata.key = "names"
    metadata.value = ",".join(class_names)
    onnx.save(onnx_model, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX 导出成功: {output_path.resolve()}")
    print(f"ONNX 大小: {size_mb:.2f} MB")
    print("输入: images [1,3,640,640], orig_target_sizes [1,2]")
    print("输出: labels, boxes, scores")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("config", "checkpoint", "labels"):
        path = getattr(args, name)
        if not path.is_file():
            print(f"错误：{name} 文件不存在: {path}", file=sys.stderr)
            return 1

    try:
        class_names = load_class_names(args.labels)
        model = build_model(args.config, args.checkpoint)
        export_model(model, args.output, class_names)
    except Exception as error:
        print(f"导出失败: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
