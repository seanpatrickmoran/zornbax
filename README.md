
# Quickstart

This repo converts VAE checkpoints between JAX/Flax/Orbax and Torch through a neutral intermediate format:

- `*.json` stores key names and tensor metadata
- `*.dat` stores raw tensor bytes

We do this to avoid an environment that sources both JAX and Torch.

## JAX Orbax → Torch

Inspect the checkpoint:

```bash
python ./zornbax/flax_orbax_neutral_checkpoint.py inspect \
  --orbax-dir ./jax-checkpoints/run_A/step_00031200
````

Export JAX EMA weights to the neutral format:

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

Convert neutral JAX-style weights into neutral Torch-style weights:

```bash
python ./zornbax/jvae_flax_to_torch_neutral.py convert \
  --src-manifest ./artifacts/neutral/jax_weights.json \
  --src-data ./artifacts/neutral/jax_weights.dat \
  --dst-manifest-out ./artifacts/neutral/torch_keys.json \
  --dst-data-out ./artifacts/neutral/torch_keys.dat
```

Build the Torch checkpoint:

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

## Torch → JAX Orbax

Export the Torch checkpoint to the neutral format:

```bash
python ./zornbax/torch_neutral_checkpoint.py export \
  --torch-checkpoint ./artifacts/checkpoints/torch_model.pt \
  --which model \
  --manifest-out ./artifacts/neutral_rev/torch_weights.json \
  --data-out ./artifacts/neutral_rev/torch_weights.dat
```

Convert neutral Torch-style weights into neutral Flax-style weights:

```bash
python ./zornbax/wanvae_torch_to_flax_neutral.py convert \
  --src-manifest ./artifacts/neutral_rev/torch_weights.json \
  --src-data ./artifacts/neutral_rev/torch_weights.dat \
  --dst-manifest-out ./artifacts/neutral_rev/flax_keys.json \
  --dst-data-out ./artifacts/neutral_rev/flax_keys.dat
```

Build the Orbax checkpoint:

```bash
python ./zornbax/flax_orbax_neutral_checkpoint.py import \
  --manifest ./artifacts/neutral_rev/flax_keys.json \
  --data ./artifacts/neutral_rev/flax_keys.dat \
  --skeleton-checkpoint ./jax-checkpoints/run_A/step_00031200 \
  --wrap-path params \
  --out-dir ./artifacts/checkpoints/jax_from_torch_step_00031200
```
