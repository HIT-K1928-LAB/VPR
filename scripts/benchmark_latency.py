#!/usr/bin/env python3
"""Benchmark latency, parameter count, and FLOPs for VPR models.

The script compares complete model stacks under the same backbone family and
input resolution. It reports:
- backbone latency (once per run)
- aggregator latency for each configuration
- parameter count (total and trainable)
- FLOPs (approximate, via the first available backend among fvcore, thop,
  and torch.profiler)

BoQ and SALAD are both supported:
- BoQ consumes the backbone feature map.
- SALAD consumes (feature_map, cls_token), so its backbone config should keep
  return_cls_token=True. The script will also enable this automatically when
  any supplied config uses SALAD.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def build_instance(spec: Dict[str, Any]):
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class"])
    params = copy.deepcopy(spec.get("params") or {})
    return cls(**params)


def normalize_backbone_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(spec)
    params = dict(normalized.get("params") or {})
    params.pop("return_cls_token", None)
    normalized["params"] = params
    return normalized


def backbone_signature(spec: Dict[str, Any]) -> str:
    return json.dumps(normalize_backbone_spec(spec), sort_keys=True, ensure_ascii=False)


def resolve_image_size(cfg: Dict[str, Any], override: Sequence[int] | None) -> Tuple[int, int]:
    if override is not None:
        if len(override) != 2:
            raise ValueError("--image_size must contain exactly two integers.")
        return int(override[0]), int(override[1])

    dm = cfg.get("datamodule", {})
    size = dm.get("val_image_size")
    if size is None:
        size = dm.get("train_image_size")
    if size is None:
        raise ValueError("Could not infer image size from config. Pass --image_size H W explicitly.")
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError(f"Invalid image size: {size}")
    return int(size[0]), int(size[1])


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def make_dummy_input(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 3, height, width, device=device)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, warmup: int, iters: int, device: torch.device) -> List[float]:
    times: List[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize(device)
                times.append(float(start.elapsed_time(end)))
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1000.0)
    return times


def stats(values: Sequence[float], batch_size: int) -> Dict[str, float]:
    mean_ms = float(statistics.mean(values))
    std_ms = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    median_ms = float(statistics.median(values))
    fps = float(batch_size * 1000.0 / mean_ms) if mean_ms > 0 else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "fps": fps,
    }


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def format_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_millions(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / 1e6:.3f}"


def format_billions(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / 1e9:.3f}"


def print_table(rows: List[List[str]]) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row_idx, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if row_idx == 0:
            print("  ".join("-" * w for w in widths))


def prepare_aggregator_input(model_output: Any, aggregator_name: str) -> Any:
    if aggregator_name.lower() == "salad":
        if not isinstance(model_output, (tuple, list)) or len(model_output) < 2:
            raise ValueError(
                "SALAD requires the backbone to return (feature_map, cls_token). "
                "Set backbone.params.return_cls_token: true in the config."
            )
        return model_output[0], model_output[1]

    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


class FullModelWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, aggregator: nn.Module, aggregator_name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.aggregator_name = aggregator_name

    def forward(self, x):
        out = self.backbone(x)
        out = prepare_aggregator_input(out, self.aggregator_name)
        return self.aggregator(out)


def count_flops(
    model: nn.Module,
    input_obj: Any,
    device: torch.device,
    backend: str,
    label: str,
) -> Optional[int]:
    if backend == "off":
        return None

    candidates = [backend] if backend != "auto" else ["fvcore", "thop", "torchprof"]
    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            if candidate == "fvcore":
                from fvcore.nn import FlopCountAnalysis

                analysis = FlopCountAnalysis(model.eval(), (input_obj,))
                total = int(analysis.total())
                if total > 0:
                    return total
                raise RuntimeError("fvcore returned zero FLOPs")

            if candidate == "thop":
                from thop import profile

                flops, _ = profile(model.eval(), inputs=(input_obj,), verbose=False)
                total = int(flops)
                if total > 0:
                    return total
                raise RuntimeError("thop returned zero FLOPs")

            if candidate == "torchprof":
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)

                with torch.profiler.profile(
                    activities=activities,
                    with_flops=True,
                    record_shapes=False,
                    profile_memory=False,
                ) as prof:
                    with torch.inference_mode():
                        model.eval()(input_obj)

                total = 0
                for event in prof.key_averages():
                    total += int(getattr(event, "flops", 0) or 0)
                if total > 0:
                    return total
                raise RuntimeError("torch.profiler returned zero FLOPs")

            raise ValueError(f"Unknown FLOPs backend: {candidate}")
        except Exception as exc:
            last_error = exc
            if backend != "auto":
                raise

    print(f"[warn] Unable to estimate FLOPs for {label}: {last_error}", file=sys.stderr)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VPR latency, params, and FLOPs")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML config files to benchmark")
    parser.add_argument("--image_size", nargs=2, type=int, default=None, help="Override input size as H W")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy batch size used for timing")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations before timing")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cuda:0, cpu, or auto")
    parser.add_argument(
        "--mode",
        type=str,
        default="aggregator",
        choices=["aggregator", "full", "both"],
        help="What to benchmark",
    )
    parser.add_argument(
        "--flops_backend",
        type=str,
        default="auto",
        choices=["auto", "fvcore", "thop", "torchprof", "off"],
        help="Backend used to estimate FLOPs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dummy input")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save results as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    config_paths = [Path(p).expanduser().resolve() for p in args.configs]
    configs = [load_yaml(p) for p in config_paths]

    ref_backbone_sig = backbone_signature(configs[0]["backbone"])
    for path, cfg in zip(config_paths[1:], configs[1:]):
        if backbone_signature(cfg["backbone"]) != ref_backbone_sig:
            raise ValueError(
                f"Backbone mismatch between {config_paths[0].name} and {path.name}. "
                "Please use configs with the same backbone family and depth."
            )

    resolved_sizes = [resolve_image_size(cfg, args.image_size) for cfg in configs]
    if args.image_size is None and any(size != resolved_sizes[0] for size in resolved_sizes[1:]):
        raise ValueError(
            "The configs resolve to different image sizes. Pass --image_size H W to benchmark them fairly."
        )
    height, width = resolved_sizes[0]

    any_salad = any(cfg["aggregator"]["class"].lower() == "salad" for cfg in configs)
    backbone_spec = copy.deepcopy(configs[0]["backbone"])
    backbone_params = dict(backbone_spec.get("params") or {})
    if any_salad:
        backbone_params["return_cls_token"] = True
    backbone_spec["params"] = backbone_params

    backbone = build_instance(backbone_spec).to(device).eval()
    backbone_total_params, backbone_trainable_params = count_parameters(backbone)

    dummy = make_dummy_input(args.batch_size, height, width, device)

    backbone_flops = count_flops(
        backbone,
        dummy,
        device,
        args.flops_backend,
        label=f"backbone {backbone_spec['class']}",
    )
    backbone_times = measure(lambda: backbone(dummy), args.warmup, args.iters, device)
    backbone_stats = stats(backbone_times, args.batch_size)

    with torch.inference_mode():
        backbone_out = backbone(dummy)

    results: List[Dict[str, Any]] = []
    rows: List[List[str]] = [[
        "config",
        "agg_params(M)",
        "total_params(M)",
        "agg_flops(G)",
        "total_flops(G)",
        "agg_ms",
        "est_full_ms",
        "full_ms",
    ]]

    print(f"Device: {device}")
    print(f"Input:  batch_size={args.batch_size}, image_size=({height}, {width})")
    print(
        f"Backbone: {backbone_spec['class']} / "
        f"{backbone_spec.get('params', {}).get('backbone_name', 'unknown')}"
    )
    print(
        f"Backbone params: {format_millions(backbone_total_params)}M total, "
        f"{format_millions(backbone_trainable_params)}M trainable"
    )
    print(f"Backbone FLOPs: {format_billions(backbone_flops)}G")
    print(f"Backbone latency: {format_ms(backbone_stats['mean_ms'])} +/- {format_ms(backbone_stats['std_ms'])} ms")

    for path, cfg in zip(config_paths, configs):
        aggregator_name = cfg["aggregator"]["class"]
        aggregator_spec = copy.deepcopy(cfg["aggregator"])
        params = aggregator_spec.get("params") or {}
        if "in_channels" in params and params["in_channels"] is None:
            params["in_channels"] = getattr(backbone, "out_channels", None)
        aggregator_spec["params"] = params

        aggregator = build_instance(aggregator_spec).to(device).eval()
        agg_total_params, _ = count_parameters(aggregator)

        agg_input = prepare_aggregator_input(backbone_out, aggregator_name)
        agg_flops = count_flops(
            aggregator,
            agg_input,
            device,
            args.flops_backend,
            label=f"aggregator {aggregator_name} ({path.stem})",
        )

        agg_times = measure(lambda: aggregator(agg_input), args.warmup, args.iters, device)
        agg_stats = stats(agg_times, args.batch_size)

        total_params = backbone_total_params + agg_total_params
        total_flops: Optional[int]
        if backbone_flops is not None and agg_flops is not None:
            total_flops = backbone_flops + agg_flops
        else:
            full_wrapper = FullModelWrapper(backbone, aggregator, aggregator_name).to(device).eval()
            total_flops = count_flops(
                full_wrapper,
                dummy,
                device,
                args.flops_backend,
                label=f"full stack ({path.stem})",
            )

        full_ms: Optional[float] = None
        if args.mode in {"full", "both"}:
            full_wrapper = FullModelWrapper(backbone, aggregator, aggregator_name).to(device).eval()
            full_times = measure(lambda: full_wrapper(dummy), args.warmup, args.iters, device)
            full_stats = stats(full_times, args.batch_size)
            full_ms = full_stats["mean_ms"]

        result: Dict[str, Any] = {
            "config": str(path),
            "backbone": backbone_spec["class"],
            "backbone_name": backbone_spec.get("params", {}).get("backbone_name"),
            "aggregator": aggregator_name,
            "image_size": [height, width],
            "batch_size": args.batch_size,
            "backbone_params_total": backbone_total_params,
            "backbone_params_trainable": backbone_trainable_params,
            "aggregator_params_total": agg_total_params,
            "total_params": total_params,
            "backbone_flops": backbone_flops,
            "aggregator_flops": agg_flops,
            "total_flops": total_flops,
            "backbone_ms": backbone_stats["mean_ms"],
            "backbone_std_ms": backbone_stats["std_ms"],
            "aggregator_ms": agg_stats["mean_ms"],
            "aggregator_std_ms": agg_stats["std_ms"],
            "estimated_full_ms": backbone_stats["mean_ms"] + agg_stats["mean_ms"],
            "full_ms": full_ms,
        }
        results.append(result)

        rows.append([
            path.stem,
            format_millions(agg_total_params),
            format_millions(total_params),
            format_billions(agg_flops),
            format_billions(total_flops),
            format_ms(agg_stats["mean_ms"]),
            format_ms(backbone_stats["mean_ms"] + agg_stats["mean_ms"]),
            format_ms(full_ms),
        ])

    print_table(rows)

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Wrote results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
