#### inspect checkpoint
python ./zornbax/flax_orbax_neutral_checkpoint.py inspect \
  --orbax-dir ./jax-checkpoints/run_A/step_00031200


#### JAX Orbax checkpoint -> neutral weights
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


#### neutral Flax-style weights -> neutral Torch-style weights
python ./zornbax/jvae_flax_to_torch_neutral.py convert \
  --src-manifest ./artifacts/neutral/jax_weights.json \
  --src-data ./artifacts/neutral/jax_weights.dat \
  --dst-manifest-out ./artifacts/neutral/torch_keys.json \
  --dst-data-out ./artifacts/neutral/torch_keys.dat


#### neutral Torch-style weights -> Torch checkpoint
python ./zornbax/torch_neutral_checkpoint.py import \
  --manifest ./artifacts/neutral/torch_keys.json \
  --data ./artifacts/neutral/torch_keys.dat \
  --arch-py ./torch_model/model/compiled_vae_model.py \
  --arch-class TorchVAECompiled \
  --arch-kwargs-json '{"dim":96,"dec_dim":96,"z_dim":16,"dim_mult":[1,2,4,4],"num_res_blocks":2,"attn_scales":[],"temperal_downsample":[false,false,false],"dropout":0.0}' \
  --output-mode model_keyed_checkpoint \
  --out ./artifacts/checkpoints/torch_model.pt
