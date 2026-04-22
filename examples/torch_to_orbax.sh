#### Torch checkpoint -> neutral weights
python ./zornbax/torch_neutral_checkpoint.py export \
  --torch-checkpoint ./artifacts/checkpoints/torch_model.pt \
  --which model \
  --manifest-out ./artifacts/neutral_rev/torch_weights.json \
  --data-out ./artifacts/neutral_rev/torch_weights.dat


#### neutral Torch-style weights -> neutral Flax-style weights
python ./zornbax/wanvae_torch_to_flax_neutral.py convert \
  --src-manifest ./artifacts/neutral_rev/torch_weights.json \
  --src-data ./artifacts/neutral_rev/torch_weights.dat \
  --dst-manifest-out ./artifacts/neutral_rev/flax_keys.json \
  --dst-data-out ./artifacts/neutral_rev/flax_keys.dat


#### neutral Flax-style weights -> Orbax checkpoint (params subtree)
python ./zornbax/flax_orbax_neutral_checkpoint.py import \
  --manifest ./artifacts/neutral_rev/flax_keys.json \
  --data ./artifacts/neutral_rev/flax_keys.dat \
  --skeleton-checkpoint ./jax-checkpoints/run_A/step_00031200 \
  --wrap-path params \
  --out-dir ./artifacts/checkpoints/jax_from_torch_step_00031200
