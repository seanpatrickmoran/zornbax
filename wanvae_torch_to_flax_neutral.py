#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np


# -----------------------------
# Neutral format I/O
# -----------------------------

def _ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _dtype_to_str(dtype: np.dtype) -> str:
    return np.dtype(dtype).str


def _str_to_dtype(s: str) -> np.dtype:
    return np.dtype(s)


def load_neutral_manifest(manifest_path: str) -> dict:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('format') != 'neutral_npmemmap_v1':
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')}")
    return manifest


def neutral_to_numpy_dict(manifest_path: str, data_path: str | None = None) -> OrderedDict[str, np.ndarray]:
    manifest = load_neutral_manifest(manifest_path)
    if data_path is None:
        data_path = os.path.join(os.path.dirname(manifest_path), manifest['data_file'])
    data = np.memmap(data_path, mode='r', dtype=np.uint8)
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    for entry in manifest['entries']:
        dt = _str_to_dtype(entry['dtype'])
        start = int(entry['offset_bytes'])
        end = start + int(entry['nbytes'])
        raw = np.frombuffer(data[start:end], dtype=dt, count=int(entry['n_elems']))
        out[entry['key']] = np.array(raw.reshape(entry['shape']), copy=True)
    return out


def export_numpy_dict_to_neutral(np_dict: Dict[str, np.ndarray], data_path: str, manifest_path: str, *, metadata: dict | None = None) -> None:
    items = []
    byte_cursor = 0
    payloads = []

    for key, arr in np_dict.items():
        arr = np.asarray(arr)
        flat = arr.reshape(-1)
        raw = flat.view(np.uint8)
        payloads.append(raw)
        items.append(
            {
                'key': key,
                'shape': list(arr.shape),
                'dtype': _dtype_to_str(arr.dtype),
                'offset_bytes': byte_cursor,
                'nbytes': int(raw.size),
                'n_elems': int(flat.size),
            }
        )
        byte_cursor += int(raw.size)

    _ensure_parent(data_path)
    _ensure_parent(manifest_path)
    data = np.memmap(data_path, mode='w+', dtype=np.uint8, shape=(byte_cursor,))
    cursor = 0
    for raw in payloads:
        n = int(raw.size)
        data[cursor: cursor + n] = raw
        cursor += n
    data.flush()

    manifest = {
        'format': 'neutral_npmemmap_v1',
        'producer': 'wanvae_torch_to_flax_neutral.py',
        'tensor_count': len(items),
        'data_file': os.path.basename(data_path),
        'entries': items,
        'metadata': metadata or {},
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


# -----------------------------
# WAN VAE mapping helpers
# -----------------------------

def _canonical_torch_key(key: str) -> str:
    parts = [p for p in key.split('.') if p and p not in {'module', '_orig_mod'}]
    return '.'.join(parts)


def _conv3d_torch_to_flax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 5:
        raise ValueError(f'Expected 5D conv3d weight, got {arr.shape}')
    return np.transpose(arr, (2, 3, 4, 1, 0))


def _conv2d_torch_to_flax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f'Expected 4D conv2d weight, got {arr.shape}')
    return np.transpose(arr, (2, 3, 1, 0))


def _gamma_squeeze(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr.reshape((arr.shape[0],)) if arr.ndim > 1 else arr


def map_torch_key_to_flax(key: str, arr: np.ndarray) -> tuple[str, np.ndarray]:
    key = _canonical_torch_key(key)

    if key == 'post_enc.weight':
        return 'post_enc/kernel', _conv3d_torch_to_flax(arr)
    if key == 'post_enc.bias':
        return 'post_enc/bias', np.asarray(arr)
    if key == 'pre_dec.weight':
        return 'pre_dec/kernel', _conv3d_torch_to_flax(arr)
    if key == 'pre_dec.bias':
        return 'pre_dec/bias', np.asarray(arr)

    m = re.fullmatch(r'(encoder|decoder)\.conv_in\.(weight|bias)', key)
    if m:
        root, leaf = m.groups()
        return f'{root}/conv_in/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)

    m = re.fullmatch(r'(encoder|decoder)\.head\.0\.gamma', key)
    if m:
        root = m.group(1)
        return f'{root}/head_norm/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'(encoder|decoder)\.head\.2\.(weight|bias)', key)
    if m:
        root, leaf = m.groups()
        return f'{root}/head_conv/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)

    m = re.fullmatch(r'(encoder|decoder)\.mid\.(0|2)\.residual\.0\.gamma', key)
    if m:
        root, idx = m.groups()
        mid = 'mid_res1' if idx == '0' else 'mid_res2'
        return f'{root}/{mid}/norm1/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'(encoder|decoder)\.mid\.(0|2)\.residual\.2\.(weight|bias)', key)
    if m:
        root, idx, leaf = m.groups()
        mid = 'mid_res1' if idx == '0' else 'mid_res2'
        return f'{root}/{mid}/conv1/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'(encoder|decoder)\.mid\.(0|2)\.residual\.3\.gamma', key)
    if m:
        root, idx = m.groups()
        mid = 'mid_res1' if idx == '0' else 'mid_res2'
        return f'{root}/{mid}/norm2/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'(encoder|decoder)\.mid\.(0|2)\.residual\.6\.(weight|bias)', key)
    if m:
        root, idx, leaf = m.groups()
        mid = 'mid_res1' if idx == '0' else 'mid_res2'
        return f'{root}/{mid}/conv2/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)

    m = re.fullmatch(r'(encoder|decoder)\.mid\.1\.norm\.gamma', key)
    if m:
        root = m.group(1)
        return f'{root}/mid_attn/norm/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'(encoder|decoder)\.mid\.1\.(to_qkv|proj)\.(weight|bias)', key)
    if m:
        root, name, leaf = m.groups()
        return f'{root}/mid_attn/{name}/{"kernel" if leaf == "weight" else "bias"}', _conv2d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)

    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.residual\.0\.gamma', key)
    if m:
        i, j = m.groups()
        return f'encoder/down_{i}/res_{j}/norm1/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.residual\.2\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'encoder/down_{i}/res_{j}/conv1/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.residual\.3\.gamma', key)
    if m:
        i, j = m.groups()
        return f'encoder/down_{i}/res_{j}/norm2/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.residual\.6\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'encoder/down_{i}/res_{j}/conv2/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.shortcut\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'encoder/down_{i}/res_{j}/shortcut/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.spatial\.1\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'encoder/down_{i}/down/spatial_conv/{"kernel" if leaf == "weight" else "bias"}', _conv2d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'encoder\.downs\.(\d+)\.blocks\.(\d+)\.time\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'encoder/down_{i}/down/time/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)

    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.residual\.0\.gamma', key)
    if m:
        i, j = m.groups()
        return f'decoder/up_{i}/res_{j}/norm1/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.residual\.2\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'decoder/up_{i}/res_{j}/conv1/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.residual\.3\.gamma', key)
    if m:
        i, j = m.groups()
        return f'decoder/up_{i}/res_{j}/norm2/gamma', _gamma_squeeze(arr)
    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.residual\.6\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'decoder/up_{i}/res_{j}/conv2/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.shortcut\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'decoder/up_{i}/res_{j}/shortcut/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.spatial\.1\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'decoder/up_{i}/up/spatial_conv/{"kernel" if leaf == "weight" else "bias"}', _conv2d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)
    m = re.fullmatch(r'decoder\.ups\.(\d+)\.blocks\.(\d+)\.time\.(weight|bias)', key)
    if m:
        i, j, leaf = m.groups()
        return f'decoder/up_{i}/up/time/{"kernel" if leaf == "weight" else "bias"}', _conv3d_torch_to_flax(arr) if leaf == 'weight' else np.asarray(arr)

    raise KeyError(f'Unmapped torch key: {key}')


# -----------------------------
# CLI
# -----------------------------

def convert_command(args: argparse.Namespace) -> None:
    src = neutral_to_numpy_dict(args.src_manifest, args.src_data)
    mapped: OrderedDict[str, np.ndarray] = OrderedDict()
    unmapped: list[str] = []

    prefix = '' if args.key_prefix is None else str(args.key_prefix).strip('/') + '/'

    for key, arr in src.items():
        try:
            new_key, new_arr = map_torch_key_to_flax(key, arr)
            new_key = prefix + new_key
            if new_key in mapped:
                raise KeyError(f'Duplicate mapped key: {new_key} from source {key}')
            mapped[new_key] = np.asarray(new_arr)
        except Exception:
            unmapped.append(key)
            if not args.allow_unmapped:
                raise

    metadata = {
        'source': 'neutral_torch_wanvae',
        'transform': 'wanvae_torch_to_flax',
        'src_manifest': os.path.abspath(args.src_manifest),
        'key_prefix': args.key_prefix,
    }
    if unmapped:
        metadata['unmapped_count'] = len(unmapped)
        metadata['unmapped_examples'] = unmapped[:20]

    export_numpy_dict_to_neutral(mapped, args.dst_data_out, args.dst_manifest_out, metadata=metadata)
    print(f'Wrote {len(mapped)} mapped tensors to neutral format:')
    print(f'  manifest: {args.dst_manifest_out}')
    print(f'  data:     {args.dst_data_out}')
    if unmapped:
        print(f'Skipped {len(unmapped)} unmapped keys (showing up to 20):')
        for k in unmapped[:20]:
            print(f'  - {k}')


def inspect_command(args: argparse.Namespace) -> None:
    manifest = load_neutral_manifest(args.manifest)
    print(json.dumps(
        {
            'format': manifest['format'],
            'tensor_count': manifest['tensor_count'],
            'data_file': manifest['data_file'],
            'metadata': manifest.get('metadata', {}),
            'first_entries': manifest['entries'][: min(10, len(manifest['entries']))],
        },
        indent=2,
    ))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='WAN VAE neutral Torch->Flax remap/transpose utility')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_convert = sub.add_parser('convert', help='Convert Torch-style neutral tensors to Flax-style neutral tensors')
    p_convert.add_argument('--src-manifest', required=True)
    p_convert.add_argument('--src-data', default=None)
    p_convert.add_argument('--dst-manifest-out', required=True)
    p_convert.add_argument('--dst-data-out', required=True)
    p_convert.add_argument('--key-prefix', default=None, help='Optional slash prefix like params to prepend to output keys')
    p_convert.add_argument('--allow-unmapped', action='store_true')
    p_convert.set_defaults(func=convert_command)

    p_inspect = sub.add_parser('inspect', help='Inspect a neutral manifest')
    p_inspect.add_argument('--manifest', required=True)
    p_inspect.set_defaults(func=inspect_command)

    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

