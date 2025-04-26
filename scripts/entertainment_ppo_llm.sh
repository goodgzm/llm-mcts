CUDA_VISIBLE_DEVICES=3 \
python twosome/virtualhome/ppo_llm_v2.py \
  --exp-name "watch_tv_ppo_llm"\
  --policy-learning-rate 1e-6 \
  --value-learning-rate 5e-5 \
  --num-envs 2 \
  --num-steps 8 \
  --policy-num-minibatches 16 \
  --value-num-minibatches 4 \
  --update-epochs 1 \
  --total-timesteps 200000 \
  --critic-warm-up-steps 0 \
  --target-kl 0.02 \
  --gradient-checkpointing-steps 8 \
  --env-id "VirtualHome-v2" \
  --record-path "workdir/origin" \
  --normalization-mode "word" \
  --gamma 0.95 \
  --seed 100 \
  --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

