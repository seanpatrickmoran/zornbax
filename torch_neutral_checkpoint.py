#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


# -----------------------------
# Generic helpers
# -----------------------------


def _load_python_object(module_path: str | None, module_name: str | None, class_name: str):
    if module_path:
        module_path = str(Path(module_path).resolve())
        stem = Path(module_path).stem
        unique_name = f"_dynamic_{stem}_{hashlib.md5(module_path.encode('utf-8')).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import module from path: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(unique_name, None)
            raise
    elif module_name:
        mod = importlib.import_module(module_name)
    else:
        raise ValueError("Need either module_path or module_name")
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise AttributeError(f"Class {class_name!r} not found") from e



def _parse_json_arg(s: str | None, default: Any) -> Any:
    if s is None:
        return default
    return json.loads(s)



def _json_load(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def _strip_wrapper_tokens(k: str) -> str:
    parts = [p for p in k.split(".") if p not in {"module", "_orig_mod"}]
    return ".".join(parts)



def _normalize_state_dict_keys(sd: Dict[str, Any]) -> OrderedDict[str, Any]:
    out: OrderedDict[str, Any] = OrderedDict()
    for k, v in sd.items():
        out[_strip_wrapper_tokens(k)] = v
    return out



def _extract_state_dict_from_torch_checkpoint(ckpt: Any, which: str = "model") -> OrderedDict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if which in ckpt and isinstance(ckpt[which], dict):
            sd = ckpt[which]
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            keys = sorted(ckpt.keys())
            raise KeyError(
                f"Could not find tensor state_dict for which={which!r}. Top-level keys: {keys}"
            )
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    sd = _normalize_state_dict_keys(sd)
    return OrderedDict((k, v.detach().cpu()) for k, v in sd.items())



def _ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)



def _dtype_to_str(dtype: np.dtype) -> str:
    return np.dtype(dtype).str



def _str_to_dtype(s: str) -> np.dtype:
    return np.dtype(s)


# -----------------------------
# Neutral format I/O
# -----------------------------


def export_state_dict_to_neutral(
    state_dict: Dict[str, torch.Tensor],
    data_path: str,
    manifest_path: str,
    *,
    metadata: dict | None = None,
) -> None:
    items = []
    byte_cursor = 0
    payloads = []

    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().contiguous().numpy()
        flat = np.asarray(arr).reshape(-1)
        raw = flat.view(np.uint8)
        payloads.append(raw)
        items.append(
            {
                "key": key,
                "shape": list(arr.shape),
                "dtype": _dtype_to_str(arr.dtype),
                "offset_bytes": byte_cursor,
                "nbytes": int(raw.size),
                "n_elems": int(flat.size),
            }
        )
        byte_cursor += int(raw.size)

    _ensure_parent(data_path)
    _ensure_parent(manifest_path)
    data = np.memmap(data_path, mode="w+", dtype=np.uint8, shape=(byte_cursor,))
    cursor = 0
    for raw in payloads:
        n = int(raw.size)
        data[cursor: cursor + n] = raw
        cursor += n
    data.flush()

    manifest = {
        "format": "neutral_npmemmap_v1",
        "producer": "torch_neutral_checkpoint.py",
        "tensor_count": len(items),
        "data_file": os.path.basename(data_path),
        "entries": items,
        "metadata": metadata or {},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)



def load_neutral_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "neutral_npmemmap_v1":
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')}")
    return manifest



def neutral_to_numpy_dict(manifest_path: str, data_path: str | None = None) -> OrderedDict[str, np.ndarray]:
    manifest = load_neutral_manifest(manifest_path)
    if data_path is None:
        data_path = os.path.join(os.path.dirname(manifest_path), manifest["data_file"])
    data = np.memmap(data_path, mode="r", dtype=np.uint8)
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    for entry in manifest["entries"]:
        dt = _str_to_dtype(entry["dtype"])
        start = int(entry["offset_bytes"])
        end = start + int(entry["nbytes"])
        raw = np.frombuffer(data[start:end], dtype=dt, count=int(entry["n_elems"]))
        out[entry["key"]] = np.array(raw.reshape(entry["shape"]), copy=True)
    return out


# -----------------------------
# Torch-side import/export
# -----------------------------


def _resolve_arch_spec(args: argparse.Namespace) -> dict:
    if args.arch_blueprint:
        return _json_load(args.arch_blueprint)
    if not args.arch_class:
        raise ValueError("Need either --arch-blueprint or --arch-py/--arch-module + --arch-class")
    return {
        "module_path": args.arch_py,
        "module_name": args.arch_module,
        "class_name": args.arch_class,
        "constructor_kwargs": _parse_json_arg(args.arch_kwargs_json, {}),
    }



def _instantiate_torch_model_from_spec(spec: dict):
    cls = _load_python_object(
        spec.get("module_path"),
        spec.get("module_name"),
        spec["class_name"],
    )
    kwargs = spec.get("constructor_kwargs", {})
    model = cls(**kwargs)
    return model, spec



def export_command(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.torch_checkpoint, map_location="cpu", weights_only=False)
    sd = _extract_state_dict_from_torch_checkpoint(ckpt, which=args.which)

    metadata = {
        "source": "torch",
        "which": args.which,
        "torch_checkpoint": os.path.abspath(args.torch_checkpoint),
    }
    export_state_dict_to_neutral(sd, args.data_out, args.manifest_out, metadata=metadata)
    print(f"Wrote {len(sd)} tensors to neutral format:")
    print(f"  manifest: {args.manifest_out}")
    print(f"  data:     {args.data_out}")



def import_command(args: argparse.Namespace) -> None:
    np_dict = neutral_to_numpy_dict(args.manifest, args.data)
    torch_sd = OrderedDict((k, torch.from_numpy(v)) for k, v in np_dict.items())

    if args.arch_blueprint or args.arch_class:
        spec = _resolve_arch_spec(args)
        model, spec = _instantiate_torch_model_from_spec(spec)
        incompat = model.load_state_dict(torch_sd, strict=not args.non_strict)
        missing = list(getattr(incompat, 'missing_keys', []))
        unexpected = list(getattr(incompat, 'unexpected_keys', []))
        print(f"Loaded into instantiated model: {spec['class_name']}")
        print(f"  missing keys:    {missing}")
        print(f"  unexpected keys: {unexpected}")
        state_dict_for_save = model.state_dict()
    else:
        state_dict_for_save = torch_sd

    out_obj: dict[str, Any] | OrderedDict[str, torch.Tensor]
    if args.output_mode == "state_dict":
        out_obj = state_dict_for_save
    elif args.output_mode == "model_keyed_checkpoint":
        out_obj = {"model": state_dict_for_save}
    elif args.output_mode == "ema_keyed_checkpoint":
        out_obj = {"ema": state_dict_for_save}
    else:
        raise ValueError(f"Unknown output_mode: {args.output_mode}")

    _ensure_parent(args.out)
    torch.save(out_obj, args.out)
    print(f"Saved torch artifact to: {args.out}")



def inspect_command(args: argparse.Namespace) -> None:
    manifest = load_neutral_manifest(args.manifest)
    print(json.dumps(
        {
            "format": manifest["format"],
            "tensor_count": manifest["tensor_count"],
            "data_file": manifest["data_file"],
            "metadata": manifest.get("metadata", {}),
            "first_entries": manifest["entries"][: min(10, len(manifest["entries"]))],
        },
        indent=2,
    ))



def add_arch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arch-blueprint", default=None, help="JSON spec for module/class/kwargs")
    p.add_argument("--arch-py", default=None, help="Path to a Python file containing the Torch class")
    p.add_argument("--arch-module", default=None, help="Importable Python module containing the Torch class")
    p.add_argument("--arch-class", default=None, help="Torch class name to instantiate")
    p.add_argument("--arch-kwargs-json", default=None, help="JSON string of constructor kwargs")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Torch <-> neutral memmap/json checkpoint utility")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export a torch checkpoint to neutral memmap/json")
    p_export.add_argument("--torch-checkpoint", required=True)
    p_export.add_argument("--which", default="model", choices=["model", "ema"])
    p_export.add_argument("--manifest-out", required=True)
    p_export.add_argument("--data-out", required=True)
    p_export.set_defaults(func=export_command)

    p_import = sub.add_parser("import", help="Import neutral memmap/json and save a torch artifact")
    p_import.add_argument("--manifest", required=True)
    p_import.add_argument("--data", default=None)
    add_arch_args(p_import)
    p_import.add_argument("--non-strict", action="store_true")
    p_import.add_argument(
        "--output-mode",
        default="model_keyed_checkpoint",
        choices=["state_dict", "model_keyed_checkpoint", "ema_keyed_checkpoint"],
    )
    p_import.add_argument("--out", required=True)
    p_import.set_defaults(func=import_command)

    p_inspect = sub.add_parser("inspect", help="Inspect neutral manifest")
    p_inspect.add_argument("--manifest", required=True)
    p_inspect.set_defaults(func=inspect_command)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

