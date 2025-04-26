CUDA_VISIBLE_DEVICES=0 \
python twosome/overcooked/ppo_llm_pomdp.py \
  --exp-name "tomato_salad_llm"\
  --policy-learning-rate 5e-7 \
  --value-learning-rate 1e-5 \
  --num-envs 4 \
  --num-steps 16 \
  --update-epochs 1 \
  --total-timesteps 50000 \
  --critic-warm-up-steps 0 \
  --env-reward 0.2 1 0.1 0.001 \
  --target-kl 0.02 \
  --gradient-checkpointing-steps 8 \
  --task 0 \
  --env-id "Overcooked-LLMA-v4" \
  --record-path "workdir/origin" \
  --normalization-mode "word" \
  --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

