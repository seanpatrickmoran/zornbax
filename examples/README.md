# Checkpoint Conversion

This repo supports moving VAE weights between JAX/Flax/Orbax checkpoints and Torch checkpoints through a neutral intermediate format:

- `*.json` for key names, shapes, dtypes, and offsets
- `*.dat` for raw tensor storage

The conversion flow is:

- Orbax checkpoint → neutral weights → Torch checkpoint
- Torch checkpoint → neutral weights → Orbax checkpoint

## Paths used below

Adjust these paths for your project layout:

- JAX checkpoint: `./jax-checkpoints/run_A/step_00031200`
- JAX model file: `./jax_model/model/jax_vae_model.py`
- Torch model file: `./torch_model/model/compiled_vae_model.py`

All generated artifacts go under `./artifacts/`.

## 1. Inspect a JAX Orbax checkpoint

Use this to see what is stored in the checkpoint before exporting weights.

```bash
python ./zornbax/flax_orbax_neutral_checkpoint.py inspect \
  --orbax-dir ./jax-checkpoints/run_A/step_00031200
````

## 2. Export JAX weights to the neutral format

This reads the `ema_params` subtree from the Orbax checkpoint and writes a neutral weight pair.

```bash
python ./zornbax/flax_orbax_neutral_checkpoint.py export \
  --orbax-dir ./jax-checkpoints/run_A/step_00031200 \
  --select-path ema_params \
  --arch-py ./jax_model/model/jax_vae_model.py \
  --arch-class JaxVAE \
  --arch-kwargs-json '{"dim":96,"dec_dim":96,"z_dim":16,"in_channels":1,"dim_mult":[1,2,4,4],"num_res_blocks":2,"attn_scales":[],"temperal_downsample":[false,false,false],"dropout":0.0}' \
  --init-rngs-json '{"params":0}' \
  --init-args-json '[{"kind":"zeros","shape":[1,4,64,64,1],"dtype":"float32"}]' \
  --init-kwargs-json '{"scale":[0.0,1.0],"sample_latent":false,"train":false}' \
  --validate-against-model \
  --manifest-out ./artifacts/neutral/jax_weights.json \
  --data-out ./artifacts/neutral/jax_weights.dat
```

## 3. Convert neutral JAX-style weights into neutral Torch-style weights

This step handles key renaming and tensor layout changes.

```bash
python ./zornbax/jvae_flax_to_torch_neutral.py convert \
  --src-manifest ./artifacts/neutral/jax_weights.json \
  --src-data ./artifacts/neutral/jax_weights.dat \
  --dst-manifest-out ./artifacts/neutral/torch_keys.json \
  --dst-data-out ./artifacts/neutral/torch_keys.dat
```

## 4. Build a Torch checkpoint

This takes the Torch-style neutral weights and materializes a real Torch checkpoint.

```bash
python ./zornbax/torch_neutral_checkpoint.py import \
  --manifest ./artifacts/neutral/torch_keys.json \
  --data ./artifacts/neutral/torch_keys.dat \
  --arch-py ./torch_model/model/compiled_vae_model.py \
  --arch-class TorchVAECompiled \
  --arch-kwargs-json '{"dim":96,"dec_dim":96,"z_dim":16,"dim_mult":[1,2,4,4],"num_res_blocks":2,"attn_scales":[],"temperal_downsample":[false,false,false],"dropout":0.0}' \
  --output-mode model_keyed_checkpoint \
  --out ./artifacts/checkpoints/torch_model.pt
```

## 5. Export a Torch checkpoint to the neutral format

To go the other direction, first export the Torch checkpoint to a neutral pair.

```bash
python ./zornbax/torch_neutral_checkpoint.py export \
  --torch-checkpoint ./artifacts/checkpoints/torch_model.pt \
  --which model \
  --manifest-out ./artifacts/neutral_rev/torch_weights.json \
  --data-out ./artifacts/neutral_rev/torch_weights.dat
```

## 6. Convert neutral Torch-style weights into neutral Flax-style weights

```bash
python ./zornbax/wanvae_torch_to_flax_neutral.py convert \
  --src-manifest ./artifacts/neutral_rev/torch_weights.json \
  --src-data ./artifacts/neutral_rev/torch_weights.dat \
  --dst-manifest-out ./artifacts/neutral_rev/flax_keys.json \
  --dst-data-out ./artifacts/neutral_rev/flax_keys.dat
```

## 7. Build an Orbax checkpoint

This writes the converted weights into a new Orbax checkpoint using an existing checkpoint as a skeleton.

```bash
python ./zornbax/flax_orbax_neutral_checkpoint.py import \
  --manifest ./artifacts/neutral_rev/flax_keys.json \
  --data ./artifacts/neutral_rev/flax_keys.dat \
  --skeleton-checkpoint ./jax-checkpoints/run_A/step_00031200 \
  --wrap-path params \
  --out-dir ./artifacts/checkpoints/jax_from_torch_step_00031200
```

## Notes

* The neutral format is only an interchange format. It is not directly loadable by Torch or Orbax without the import step.
* The Flax↔Torch conversion scripts are model-specific. They assume the JAX and Torch architectures correspond.
* If you want to convert EMA weights instead of online weights on the Torch side, use `--which ema` in the Torch export step.
