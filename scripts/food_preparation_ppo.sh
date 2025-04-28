CUDA_VISIBLE_DEVICES=0 \
python /data/gzm/TWOSOME-main/twosome_rl/vitualhome/ppo_rl.py \
  --env-id 'VirtualHome-v1' \
  --exp-name "heat_pancake_ppo_llm" \
  --hid 128 \
  --l 2 \
  --gamma 0.99 \
  --seed 10 \
  --steps 5000 \
  --epochs 100 \
  --ent-coef 0.01 \
  --norm-adv False \
  --track True \
  --wandb-project-name "twosome_ppo" \
  --cuda False 