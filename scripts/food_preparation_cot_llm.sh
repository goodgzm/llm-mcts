CUDA_VISIBLE_DEVICES=1 \
python twosome-CotTot/virtualhome/tot_policy_pomdp_inference_v1.py \
  --exp-name "heat_pancake_ppo_llm"\
  --num-envs 1 \
  --env-id "VirtualHome-v1" \
  --record-path "workdir/food_preparation/tot" \
  --normalization-mode "word" \
  --stochastic 0 \
  --path-num 2000 \
  --depth 15 \
  --llm-base-model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --llm-base-model-path "/data/gzm/TWOSOME-main/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


