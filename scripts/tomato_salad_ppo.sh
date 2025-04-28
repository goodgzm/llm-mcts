CUDA_VISIBLE_DEVICES=0 \
python twosome_rl/ppo.py \
  --exp-name "tomato_salad_ppo"\
  --policy-learning-rate 5e-7 \
  --value-learning-rate 1e-5 \
  --num-envs 4 \
  --num-steps 128 \
  --update-epochs 1 \
  --total-timesteps 1000000 \
  --critic-warm-up-steps 0 \
  --env-reward 0.2 1 0.1 0.001 \
  --target-kl 0.02 \
  --gradient-checkpointing-steps 8 \
  --task 0 \
  --env-id "Overcooked-LLMA-v4" \
  --record-path "workdir/origin" \
  --update-epochs 50\
  # --cuda false \


#   --normalization-mode "word" \
#   --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#   --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

