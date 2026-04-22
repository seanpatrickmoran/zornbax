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
from typing import Any, Mapping, MutableMapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import linen as nn
from flax import traverse_util


# -----------------------------
# Generic helpers
# -----------------------------

def _json_load(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_arg(text: str | None, default: Any) -> Any:
    if text is None:
        return default
    return json.loads(text)


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


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _dtype_to_str(dtype: np.dtype) -> str:
    return np.dtype(dtype).str


def _str_to_dtype(s: str) -> np.dtype:
    return np.dtype(s)


# -----------------------------
# Tree helpers (adapted from your JAX harness/inflate utilities)
# -----------------------------

def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _looks_like_param_leaf(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _is_param_leaf_name(name: str) -> bool:
    return name in {"kernel", "bias", "gamma", "scale", "embedding"}


def _count_param_leaves(tree: Any) -> int:
    if not _is_mapping(tree):
        return 0
    total = 0
    for k, v in tree.items():
        if _is_param_leaf_name(str(k)) and _looks_like_param_leaf(v):
            total += 1
        elif _is_mapping(v):
            total += _count_param_leaves(v)
    return total


def _get_subtree(tree: Any, path: Sequence[str]) -> Any:
    out = tree
    for key in path:
        if not _is_mapping(out) or key not in out:
            raise KeyError("/".join(path))
        out = out[key]
    return out


def _set_subtree(tree: MutableMapping[str, Any], path: Sequence[str], value: Any) -> MutableMapping[str, Any]:
    if not path:
        if not isinstance(value, MutableMapping):
            raise TypeError("Root replacement requires a mutable mapping.")
        return value
    cur = tree
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = dict(nxt) if isinstance(nxt, Mapping) else {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value
    return tree


def find_named_paths(tree: Any, wanted: str) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []

    def visit(node: Any, path: tuple[str, ...]):
        if not _is_mapping(node):
            return
        if path and path[-1] == wanted:
            out.append(path)
        for k, v in node.items():
            visit(v, path + (str(k),))

    visit(tree, ())
    return out


def find_param_subtree_candidates(tree: Any) -> list[tuple[tuple[str, ...], int]]:
    candidates: list[tuple[tuple[str, ...], int]] = []

    def visit(node: Any, path: tuple[str, ...]):
        if not _is_mapping(node):
            return
        count = _count_param_leaves(node)
        if count > 0 and (path or any(_is_param_leaf_name(str(k)) for k in node.keys())):
            candidates.append((path, count))
        for k, v in node.items():
            visit(v, path + (str(k),))

    visit(tree, ())
    return sorted(candidates, key=lambda x: (-x[1], len(x[0]), x[0]))


def find_params_path(tree: Any, preferred: str = "auto") -> tuple[str, ...]:
    if preferred != "auto":
        return tuple(x for x in preferred.split("/") if x)

    named = find_named_paths(tree, "params")
    if named:
        ranked = []
        for path in named:
            subtree = _get_subtree(tree, path)
            ranked.append((path, _count_param_leaves(subtree)))
        ranked.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
        return ranked[0][0]

    candidates = find_param_subtree_candidates(tree)
    if not candidates:
        raise ValueError(
            "Could not automatically find a parameter subtree. "
            "Pass --source-params-path explicitly, e.g. 'params' or 'state/params'."
        )
    return candidates[0][0]


def summarize_tree(node: Any, depth: int = 0, max_depth: int = 3, max_items: int = 12) -> Any:
    if _looks_like_param_leaf(node):
        return {"shape": list(node.shape), "dtype": str(node.dtype)}
    if not _is_mapping(node):
        return str(type(node).__name__)
    if depth >= max_depth:
        return {"<truncated>": f"{len(node)} keys"}
    out = {}
    for i, (k, v) in enumerate(node.items()):
        if i >= max_items:
            out["<more>"] = f"{len(node) - max_items} more keys"
            break
        out[str(k)] = summarize_tree(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
    return out


def _flatten_params_to_slash_dict(tree: Mapping[str, Any], *, key_prefix: str | None = None) -> OrderedDict[str, np.ndarray]:
    flat = traverse_util.flatten_dict(tree, keep_empty_nodes=False)
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    skipped: list[tuple[str, str]] = []
    for path, value in sorted(flat.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        key = "/".join(str(x) for x in path)
        if key_prefix:
            key = f"{key_prefix.rstrip('/')}/{key}"
        if value is None:
            skipped.append((key, "None"))
            continue
        try:
            arr = np.asarray(jax.device_get(value))
        except Exception as e:
            skipped.append((key, f"{type(value).__name__}: {e}"))
            continue
        if arr.dtype == object:
            skipped.append((key, "dtype=object"))
            continue
        out[key] = arr
    if skipped:
        print(f"[flatten] skipped {len(skipped)} non-array leaves")
        for k, why in skipped[:20]:
            print(f"  - {k}: {why}")
    return out


# -----------------------------
# Neutral memmap/json I/O
# -----------------------------

def export_numpy_dict_to_neutral(np_dict: Mapping[str, np.ndarray], data_path: str, manifest_path: str, *, metadata: dict | None = None) -> None:
    items = []
    byte_cursor = 0
    payloads = []
    for key, arr in np_dict.items():
        arr = np.asarray(arr)
        flat = arr.reshape(-1)
        raw = flat.view(np.uint8)
        payloads.append(raw)
        items.append({
            "key": key,
            "shape": list(arr.shape),
            "dtype": _dtype_to_str(arr.dtype),
            "offset_bytes": byte_cursor,
            "nbytes": int(raw.size),
            "n_elems": int(flat.size),
        })
        byte_cursor += int(raw.size)

    _ensure_parent(data_path)
    _ensure_parent(manifest_path)
    mm = np.memmap(data_path, mode="w+", dtype=np.uint8, shape=(byte_cursor,))
    cursor = 0
    for raw in payloads:
        n = int(raw.size)
        mm[cursor: cursor + n] = raw
        cursor += n
    mm.flush()

    manifest = {
        "format": "neutral_npmemmap_v1",
        "producer": "flax_orbax_neutral_checkpoint_refactor.py",
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
# Flax model target helpers
# -----------------------------

def _build_value_from_spec(spec: Any) -> Any:
    if isinstance(spec, dict) and "kind" in spec:
        kind = spec["kind"]
        if kind == "array":
            shape = tuple(spec["shape"])
            dtype = getattr(jnp, spec.get("dtype", "float32"))
            fill = spec.get("fill", 0.0)
            return jnp.full(shape, fill, dtype=dtype)
        if kind == "zeros":
            shape = tuple(spec["shape"])
            dtype = getattr(jnp, spec.get("dtype", "float32"))
            return jnp.zeros(shape, dtype=dtype)
        if kind == "ones":
            shape = tuple(spec["shape"])
            dtype = getattr(jnp, spec.get("dtype", "float32"))
            return jnp.ones(shape, dtype=dtype)
        if kind == "scalar":
            return spec["value"]
        raise ValueError(f"Unknown spec kind: {kind}")
    if isinstance(spec, list):
        return [_build_value_from_spec(x) for x in spec]
    if isinstance(spec, dict):
        return {k: _build_value_from_spec(v) for k, v in spec.items()}
    return spec


def instantiate_flax_params_from_args(args):
    if args.arch_blueprint:
        bp = _json_load(args.arch_blueprint)
        module_path = bp.get("module_path")
        module_name = bp.get("module_name")
        class_name = bp["class_name"]
        constructor_kwargs = bp.get("constructor_kwargs", {})
        rng_specs = bp.get("rngs", {"params": 0})
        init_args_specs = bp.get("init_args", [])
        init_kwargs_specs = bp.get("init_kwargs", {})
    else:
        if not args.arch_class:
            raise ValueError("Need either --arch-blueprint or --arch-py/--arch-module + --arch-class")
        module_path = args.arch_py
        module_name = args.arch_module
        class_name = args.arch_class
        constructor_kwargs = _json_arg(args.arch_kwargs_json, {})
        rng_specs = _json_arg(args.init_rngs_json, {"params": 0})
        init_args_specs = _json_arg(args.init_args_json, [])
        init_kwargs_specs = _json_arg(args.init_kwargs_json, {})

    cls = _load_python_object(module_path, module_name, class_name)
    model = cls(**constructor_kwargs)
    rngs = {name: jax.random.PRNGKey(int(seed)) for name, seed in rng_specs.items()}
    init_args = [_build_value_from_spec(x) for x in init_args_specs]
    init_kwargs = {k: _build_value_from_spec(v) for k, v in init_kwargs_specs.items()}
    variables = model.init(rngs, *init_args, **init_kwargs)
    return variables["params"], {
        "module_path": module_path,
        "module_name": module_name,
        "class_name": class_name,
        "constructor_kwargs": constructor_kwargs,
        "rngs": rng_specs,
        "init_args": init_args_specs,
        "init_kwargs": init_kwargs_specs,
    }


def compare_tree_leaves(selected: Mapping[str, Any], target: Mapping[str, Any]):
    sel = _flatten_params_to_slash_dict(selected)
    tgt = _flatten_params_to_slash_dict(target)
    sel_keys = set(sel.keys())
    tgt_keys = set(tgt.keys())
    missing = sorted(tgt_keys - sel_keys)
    unexpected = sorted(sel_keys - tgt_keys)
    mismatched = []
    for k in sorted(sel_keys & tgt_keys):
        if tuple(sel[k].shape) != tuple(tgt[k].shape):
            mismatched.append((k, tuple(sel[k].shape), tuple(tgt[k].shape)))
    return missing, unexpected, mismatched


def _print_tree_diffs(missing, unexpected, mismatched, max_items: int = 40):
    if missing:
        print(f"Missing keys relative to target ({len(missing)}):")
        for k in missing[:max_items]:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys relative to target ({len(unexpected)}):")
        for k in unexpected[:max_items]:
            print(f"  {k}")
    if mismatched:
        print(f"Shape mismatches ({len(mismatched)}):")
        for k, got, want in mismatched[:max_items]:
            print(f"  {k}: selected {got} vs target {want}")


# -----------------------------
# Orbax helpers
# -----------------------------

def restore_checkpoint_tree(path: str | Path, target: dict | None = None):
    ckpt_path = Path(path).expanduser().resolve() if "://" not in str(path) else path
    checkpointer = ocp.StandardCheckpointer()
    if target is None:
        return checkpointer.restore(str(ckpt_path))
    try:
        return checkpointer.restore(str(ckpt_path), target=target)
    except TypeError:
        return checkpointer.restore(str(ckpt_path), target)


def save_checkpoint_tree(out_dir: str | Path, tree: Mapping[str, Any], *, force: bool = True):
    ckpt_path = Path(out_dir).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(ckpt_path), jax.device_get(tree), force=force)
    try:
        checkpointer.wait_until_finished()
    except AttributeError:
        pass


# -----------------------------
# Commands
# -----------------------------

def inspect_command(args):
    tree = restore_checkpoint_tree(args.orbax_dir)
    print("[inspect] top-level keys:", list(tree.keys()) if isinstance(tree, Mapping) else type(tree).__name__)
    print(json.dumps(summarize_tree(tree, max_depth=int(args.inspect_max_depth)), indent=2, sort_keys=True))
    named_params = ["/".join(p) if p else "<root>" for p in find_named_paths(tree, "params")]
    named_ema = ["/".join(p) if p else "<root>" for p in find_named_paths(tree, "ema_params")]
    print("[inspect] named params paths:", named_params)
    print("[inspect] named ema_params paths:", named_ema)
    candidates = find_param_subtree_candidates(tree)
    print("[inspect] parameter subtree candidates:")
    for path, count in candidates[:20]:
        print(f"  {'/'.join(path) if path else '<root>'}: {count} param leaves")


def _select_export_subtree(tree: Mapping[str, Any], args) -> tuple[Mapping[str, Any], str]:
    if args.select_path:
        path = tuple(x for x in args.select_path.split('/') if x)
        return _get_subtree(tree, path), '/'.join(path)

    if bool(args.prefer_ema) and isinstance(tree, Mapping) and "ema_params" in tree and tree["ema_params"] is not None:
        return tree["ema_params"], "ema_params"

    if isinstance(tree, Mapping) and "params" in tree and tree["params"] is not None:
        return tree["params"], "params"

    path = find_params_path(tree, preferred=args.source_params_path)
    return _get_subtree(tree, path), ('/'.join(path) if path else '<root>')


def export_command(args):
    tree = restore_checkpoint_tree(args.orbax_dir)
    if isinstance(tree, Mapping):
        print(f"[export] Raw restore top-level keys: {list(tree.keys())}")
    selected, selected_name = _select_export_subtree(tree, args)
    print(f"[export] selected subtree: {selected_name}")

    target_params = None
    arch_meta = None
    if args.validate_against_model:
        target_params, arch_meta = instantiate_flax_params_from_args(args)
        missing, unexpected, mismatched = compare_tree_leaves(selected, target_params)
        if missing or unexpected or mismatched:
            _print_tree_diffs(missing, unexpected, mismatched)
            raise ValueError("Selected params subtree does not match target model params.")
        print("[export] validation against model params: OK")

    flat = _flatten_params_to_slash_dict(selected, key_prefix=args.key_prefix)
    metadata = {
        "source": "orbax_flax",
        "orbax_dir": str(Path(args.orbax_dir).expanduser().resolve()),
        "selected_subtree": selected_name,
        "prefer_ema": bool(args.prefer_ema),
    }
    if arch_meta is not None:
        metadata["arch_spec"] = arch_meta
    export_numpy_dict_to_neutral(flat, args.data_out, args.manifest_out, metadata=metadata)
    print(f"Wrote {len(flat)} arrays to neutral format:")
    print(f"  manifest: {args.manifest_out}")
    print(f"  data:     {args.data_out}")


def import_command(args):
    flat = neutral_to_numpy_dict(args.manifest, args.data)
    nested = traverse_util.unflatten_dict({tuple(k.split('/')): jnp.asarray(v) for k, v in flat.items()})
    tree: MutableMapping[str, Any] = {}
    if args.wrap_path:
        path = tuple(x for x in args.wrap_path.split('/') if x)
        tree = _set_subtree({}, path, nested)
    else:
        tree = dict(nested)

    if args.skeleton_checkpoint:
        skeleton = restore_checkpoint_tree(args.skeleton_checkpoint)
        if args.wrap_path:
            tree = _set_subtree(dict(skeleton), tuple(x for x in args.wrap_path.split('/') if x), nested)
        else:
            tree = dict(skeleton)
            if args.replace_root:
                tree = dict(nested)
            else:
                raise ValueError("Use --wrap-path with --skeleton-checkpoint, or pass --replace-root to overwrite the full tree.")

    save_checkpoint_tree(args.out_dir, tree, force=True)
    print(f"Saved Orbax checkpoint to: {args.out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="Flax/Orbax <-> neutral memmap/json checkpoint utility (refactored for WAN JAX training checkpoints)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_arch_args(sp):
        sp.add_argument("--arch-blueprint", default=None)
        sp.add_argument("--arch-py", default=None)
        sp.add_argument("--arch-module", default=None)
        sp.add_argument("--arch-class", default=None)
        sp.add_argument("--arch-kwargs-json", default=None)
        sp.add_argument("--init-rngs-json", default=None)
        sp.add_argument("--init-args-json", default=None)
        sp.add_argument("--init-kwargs-json", default=None)

    p_inspect = sub.add_parser("inspect", help="Inspect an Orbax checkpoint tree")
    p_inspect.add_argument("--orbax-dir", required=True)
    p_inspect.add_argument("--inspect-max-depth", type=int, default=3)
    p_inspect.set_defaults(func=inspect_command)

    p_export = sub.add_parser("export", help="Export a selected Orbax params subtree to neutral memmap/json")
    p_export.add_argument("--orbax-dir", required=True)
    p_export.add_argument("--select-path", default=None, help="Slash path inside checkpoint tree, e.g. ema_params or params")
    p_export.add_argument("--prefer-ema", action=argparse.BooleanOptionalAction, default=True)
    p_export.add_argument("--source-params-path", default="auto", help="Fallback auto-search override, e.g. params or state/params")
    p_export.add_argument("--key-prefix", default=None, help="Optional slash prefix to prepend to exported keys")
    add_arch_args(p_export)
    p_export.add_argument("--validate-against-model", action="store_true")
    p_export.add_argument("--manifest-out", required=True)
    p_export.add_argument("--data-out", required=True)
    p_export.set_defaults(func=export_command)

    p_import = sub.add_parser("import", help="Import neutral memmap/json and save Orbax checkpoint")
    p_import.add_argument("--manifest", required=True)
    p_import.add_argument("--data", default=None)
    p_import.add_argument("--wrap-path", default=None, help="Wrap imported params under a slash path, e.g. ema_params or params")
    p_import.add_argument("--skeleton-checkpoint", default=None, help="Optional existing checkpoint tree to copy and replace a subtree into")
    p_import.add_argument("--replace-root", action="store_true")
    p_import.add_argument("--out-dir", required=True)
    p_import.set_defaults(func=import_command)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

