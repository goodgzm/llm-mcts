CUDA_VISIBLE_DEVICES=0 \
python /data/gzm/TWOSOME-main/twosome_rl/overcook/ppo_rl.py \
  --env-id 'Overcooked-LLMA-v4' \
  --task 3 \
  --map-type "A" \
  --exp-name "tomato_lettuce_salad_ppo" \
  --hid 128 \
  --l 3 \
  --gamma 0.99 \
  --seed 10 \
  --steps 5000 \
  --epochs 1000 \
  --env-reward 0.2 1 0 0.001 \
  --ent-coef 0.01 \
  --norm-adv False \
  --track True \
  --wandb-project-name "twosome_ppo" \
  --cuda False \
  # --cuda false \


#   --normalization-mode "word" \
#   --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#   --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

