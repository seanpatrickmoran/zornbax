#### inspect checkpoint
python ./zornbax/flax_orbax_neutral_checkpoint.py inspect   --orbax-dir ./jax-checkpoint-dir/jax-run-01/step_00031200

#### unroll orbax ckpt to orbaxweights
python ./zornbax/flax_orbax_neutral_checkpoint.py export \
  --orbax-dir ./jax-checkpoint-dir/jax-run-01/step_00031200 \
  --select-path ema_params \
  --arch-py ./jax_vae/model/jax_vae_model.py \
  --arch-class jVAE \
  --arch-kwargs-json '{"dim":96,"dec_dim":96,"z_dim":16,"in_channels":1,"dim_mult":[1,2,4,4],"num_res_blocks":2,"attn_scales":[],"temperal_downsample":[false,false,false],"dropout":0.0}' \
  --init-rngs-json '{"params":0}' \
  --init-args-json '[{"kind":"zeros","shape":[1,4,64,64,1],"dtype":"float32"}]' \
  --init-kwargs-json '{"scale":[0.0,1.0],"sample_latent":false,"train":false}' \
  --validate-against-model \
  --manifest-out ../zornbax_crossovers/jax_step_00031200_jaxvae_weights.json \
  --data-out ../zornbax_crossovers/jax_step_00031200_jaxvae_weights.dat

#### reroll orbaxweights to torchweights
python ~/utils/jvae_flax_to_torch_neutral.py convert \
  --src-manifest ../zornbax_crossovers/jax_step_00031200_jaxvae_weights.json \
  --src-data ../zornbax_crossovers/jax_step_00031200_jaxvae_weights.dat \
  --dst-manifest-out ../zornbax_crossovers/step_00031200_vae_torchkeys.json \
  --dst-data-out ../zornbax_crossovers/step_00031200_vae_torchkeys.dat

#### make torchfile checkpoint
python ~/zornbax/torch_neutral_checkpoint.py import \
  --manifest ../zornbax_crossovers/step_00031200_vae_torchkeys.json \
  --data ../zornbax_crossovers/step_00031200_vae_torchkeys.dat \
  --arch-py /home/spmoran/P3-jvae-hic/model/compiled_vae_model.py \
  --arch-class VAE_Compiled \
  --arch-kwargs-json '{"dim":96,"dec_dim":96,"z_dim":16,"dim_mult":[1,2,4,4],"num_res_blocks":2,"attn_scales":[],"temperal_downsample":[false,false,false],"dropout":0.0}' \
  --output-mode model_keyed_checkpoint \
  --out ../zornbax_crossovers/torch_step_00031200_vae_model.pt
