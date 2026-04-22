#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

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
        'producer': 'wanvae_flax_to_torch_neutral.py',
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


MODEL_ROOTS = {'encoder', 'decoder', 'post_enc', 'pre_dec'}



def _canonical_tokens(key: str) -> list[str]:
    tokens = [t for t in key.split('/') if t]
    if 'params' in tokens:
        tokens = tokens[tokens.index('params') + 1:]
    else:
        for i, tok in enumerate(tokens):
            if tok in MODEL_ROOTS:
                tokens = tokens[i:]
                break
    return tokens



def _parse_index(token: str, prefix: str) -> int:
    if not token.startswith(prefix):
        raise ValueError(f'Expected token starting with {prefix!r}, got {token!r}')
    return int(token[len(prefix):])



def _infer_block_counts(flat: Dict[str, np.ndarray]):
    enc_down = defaultdict(set)
    dec_up = defaultdict(set)
    for key in flat.keys():
        toks = _canonical_tokens(key)
        if len(toks) >= 3 and toks[0] == 'encoder' and toks[1].startswith('down_') and toks[2].startswith('res_'):
            enc_down[_parse_index(toks[1], 'down_')].add(_parse_index(toks[2], 'res_'))
        if len(toks) >= 3 and toks[0] == 'decoder' and toks[1].startswith('up_') and toks[2].startswith('res_'):
            dec_up[_parse_index(toks[1], 'up_')].add(_parse_index(toks[2], 'res_'))
    enc_down = {k: (max(v) + 1) for k, v in enc_down.items()}
    dec_up = {k: (max(v) + 1) for k, v in dec_up.items()}
    return enc_down, dec_up



def _conv3d_flax_to_torch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 5:
        raise ValueError(f'Expected 5D conv3d kernel, got shape {arr.shape}')
    return np.transpose(arr, (4, 3, 0, 1, 2))



def _conv2d_flax_to_torch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f'Expected 4D conv2d kernel, got shape {arr.shape}')
    return np.transpose(arr, (3, 2, 0, 1))



def _gamma_expand(arr: np.ndarray, spatial_dims: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape((arr.shape[0],) + (1,) * spatial_dims)
    return arr



def _join_torch_name(prefix: str, name: str, suffix: str) -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    if name:
        parts.append(name)
    parts.append(suffix)
    return '.'.join(parts)



def _map_conv_param(prefix: str, name: str, leaf: str, arr: np.ndarray, *, conv_ndim: int):
    if leaf == 'kernel':
        if conv_ndim == 3:
            return _join_torch_name(prefix, name, 'weight'), _conv3d_flax_to_torch(arr)
        if conv_ndim == 2:
            return _join_torch_name(prefix, name, 'weight'), _conv2d_flax_to_torch(arr)
        raise ValueError(f'Unsupported conv_ndim {conv_ndim}')
    if leaf == 'bias':
        return _join_torch_name(prefix, name, 'bias'), np.asarray(arr)
    raise KeyError(f'Unexpected conv leaf {leaf!r}')



def _map_residual(prefix: str, param_name: str, leaf: str, arr: np.ndarray):
    if param_name == 'norm1' and leaf == 'gamma':
        return f'{prefix}.residual.0.gamma', _gamma_expand(arr, 3)
    if param_name == 'conv1':
        return _map_conv_param(prefix, 'residual.2', leaf, arr, conv_ndim=3)
    if param_name == 'norm2' and leaf == 'gamma':
        return f'{prefix}.residual.3.gamma', _gamma_expand(arr, 3)
    if param_name == 'conv2':
        return _map_conv_param(prefix, 'residual.6', leaf, arr, conv_ndim=3)
    if param_name == 'shortcut':
        return _map_conv_param(prefix, 'shortcut', leaf, arr, conv_ndim=3)
    raise KeyError(f'Unexpected residual mapping component: {param_name}/{leaf}')



def _map_attention(prefix: str, param_name: str, leaf: str, arr: np.ndarray):
    if param_name == 'norm' and leaf == 'gamma':
        return f'{prefix}.norm.gamma', _gamma_expand(arr, 2)
    if param_name == 'to_qkv':
        return _map_conv_param(prefix, 'to_qkv', leaf, arr, conv_ndim=2)
    if param_name == 'proj':
        return _map_conv_param(prefix, 'proj', leaf, arr, conv_ndim=2)
    raise KeyError(f'Unexpected attention mapping component: {param_name}/{leaf}')



def map_flax_key_to_torch(key: str, arr: np.ndarray, enc_down_counts: dict[int, int], dec_up_counts: dict[int, int]):
    toks = _canonical_tokens(key)
    if not toks:
        raise KeyError(f'Empty key after canonicalization: {key}')

    root = toks[0]

    if root in ('post_enc', 'pre_dec'):
        if len(toks) != 2:
            raise KeyError(f'Unexpected top-level key structure for {key}')
        leaf = toks[1]
        return _map_conv_param('', root, leaf, arr, conv_ndim=3)

    if root == 'encoder':
        if toks[1] == 'conv_in':
            return _map_conv_param('encoder', 'conv_in', toks[2], arr, conv_ndim=3)
        if toks[1].startswith('down_'):
            down_idx = _parse_index(toks[1], 'down_')
            if toks[2].startswith('res_'):
                res_idx = _parse_index(toks[2], 'res_')
                prefix = f'encoder.downs.{down_idx}.blocks.{res_idx}'
                return _map_residual(prefix, toks[3], toks[4], arr)
            if toks[2] == 'down':
                block_idx = enc_down_counts[down_idx]
                if toks[3] == 'spatial_conv':
                    return _map_conv_param(f'encoder.downs.{down_idx}.blocks.{block_idx}', 'spatial.1', toks[4], arr, conv_ndim=2)
                if toks[3] == 'time':
                    return _map_conv_param(f'encoder.downs.{down_idx}.blocks.{block_idx}', 'time', toks[4], arr, conv_ndim=3)
                raise KeyError(f'Unexpected encoder down resample component in {key}')
        if toks[1] == 'mid_res1':
            return _map_residual('encoder.mid.0', toks[2], toks[3], arr)
        if toks[1] == 'mid_attn':
            return _map_attention('encoder.mid.1', toks[2], toks[3], arr)
        if toks[1] == 'mid_res2':
            return _map_residual('encoder.mid.2', toks[2], toks[3], arr)
        if toks[1] == 'head_norm' and toks[2] == 'gamma':
            return 'encoder.head.0.gamma', _gamma_expand(arr, 3)
        if toks[1] == 'head_conv':
            return _map_conv_param('encoder.head', '2', toks[2], arr, conv_ndim=3)

    if root == 'decoder':
        if toks[1] == 'conv_in':
            return _map_conv_param('decoder', 'conv_in', toks[2], arr, conv_ndim=3)
        if toks[1].startswith('up_'):
            up_idx = _parse_index(toks[1], 'up_')
            if toks[2].startswith('res_'):
                res_idx = _parse_index(toks[2], 'res_')
                prefix = f'decoder.ups.{up_idx}.blocks.{res_idx}'
                return _map_residual(prefix, toks[3], toks[4], arr)
            if toks[2] == 'up':
                block_idx = dec_up_counts[up_idx]
                if toks[3] == 'spatial_conv':
                    return _map_conv_param(f'decoder.ups.{up_idx}.blocks.{block_idx}', 'spatial.1', toks[4], arr, conv_ndim=2)
                if toks[3] == 'time':
                    return _map_conv_param(f'decoder.ups.{up_idx}.blocks.{block_idx}', 'time', toks[4], arr, conv_ndim=3)
                raise KeyError(f'Unexpected decoder up resample component in {key}')
        if toks[1] == 'mid_res1':
            return _map_residual('decoder.mid.0', toks[2], toks[3], arr)
        if toks[1] == 'mid_attn':
            return _map_attention('decoder.mid.1', toks[2], toks[3], arr)
        if toks[1] == 'mid_res2':
            return _map_residual('decoder.mid.2', toks[2], toks[3], arr)
        if toks[1] == 'head_norm' and toks[2] == 'gamma':
            return 'decoder.head.0.gamma', _gamma_expand(arr, 3)
        if toks[1] == 'head_conv':
            return _map_conv_param('decoder.head', '2', toks[2], arr, conv_ndim=3)

    raise KeyError(f'Unmapped key: {key}')


# -----------------------------
# CLI
# -----------------------------


def convert_command(args: argparse.Namespace) -> None:
    src = neutral_to_numpy_dict(args.src_manifest, args.src_data)
    enc_down_counts, dec_up_counts = _infer_block_counts(src)

    mapped: OrderedDict[str, np.ndarray] = OrderedDict()
    unmapped: list[str] = []
    for key, arr in src.items():
        try:
            new_key, new_arr = map_flax_key_to_torch(key, arr, enc_down_counts, dec_up_counts)
            if new_key in mapped:
                raise KeyError(f'Duplicate mapped key: {new_key} from source {key}')
            mapped[new_key] = np.asarray(new_arr)
        except Exception:
            unmapped.append(key)
            if not args.allow_unmapped:
                raise

    metadata = {
        'source': 'neutral_flax_wanvae',
        'transform': 'wanvae_flax_to_torch',
        'src_manifest': os.path.abspath(args.src_manifest),
        'encoder_down_block_counts': enc_down_counts,
        'decoder_up_block_counts': dec_up_counts,
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
    p = argparse.ArgumentParser(description='WAN VAE neutral Flax->Torch remap/transpose utility')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_convert = sub.add_parser('convert', help='Convert Flax-style neutral tensors to Torch-style neutral tensors')
    p_convert.add_argument('--src-manifest', required=True)
    p_convert.add_argument('--src-data', default=None)
    p_convert.add_argument('--dst-manifest-out', required=True)
    p_convert.add_argument('--dst-data-out', required=True)
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

